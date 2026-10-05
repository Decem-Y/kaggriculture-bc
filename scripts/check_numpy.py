"""Check that the numpy policy matches the torch model, and time one numpy step.

Compares the encoder output and the per-unit raw actions of greedy decoding (no legality mask, topk 0) on states
sampled from census episodes.

  python scripts/check_numpy.py runs/specialize/best.pt --census data/specialize/census.jsonl --team "<TEAM>" --date 41
"""
import argparse, json, os, sys, tempfile, time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import raw_from_ep, featurize, UNIT_OPS  # noqa: E402
from export import export  # noqa: E402
from model import BCNet  # noqa: E402
from policy import BCPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--census", required=True)
    ap.add_argument("--ep-dir", default=os.path.join(ROOT, "cache/ep"))
    ap.add_argument("--team", required=True)
    ap.add_argument("--date", type=int, required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = BCNet(len(ck["vocab"]), **ck["cfg"]["model"])
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    m.eval()
    d = tempfile.mkdtemp()
    export(a.ckpt, d)
    pol = BCPolicy(d, team=a.team, won=1, date=a.date, mask=False)
    rows = [json.loads(l) for l in open(a.census)]
    rng = np.random.default_rng(0)
    same = tot = 0
    hd = []
    for k in range(a.n):
        r = rows[int(rng.integers(len(rows)))]
        ep = dict(np.load(os.path.join(a.ep_dir, f"{r['id']}.npz")))
        s = int(rng.integers(2))
        raw = raw_from_ep(ep, np.array([int(rng.integers(719))]), np.array([s]))
        x = featurize(raw, np.array([pol.team]), np.array([1]), np.array([a.date]))
        xt = {kk: torch.from_numpy(v) for kk, v in x.items()}
        with torch.no_grad():
            dd, _ = m.decode(xt)
            h_t = m.encode(xt).numpy()
        h_n = pol.net.encode(x)
        hd.append(np.abs(h_t - h_n).max())
        for i, u in enumerate(pol._units(raw, h_n)):
            tot += 1
            same += u[0] == UNIT_OPS[int(dd["raw"][0, i])]
    print(f"encoder max |torch - numpy| {max(hd):.2e}; raw actions agree {same}/{tot}")
    pol = BCPolicy(d, team=a.team, won=1, date=a.date, topk=8)
    raw = raw_from_ep(ep, np.array([300]), np.array([s]))
    x = featurize(raw, np.array([pol.team]), np.array([1]), np.array([a.date]))
    t0 = time.perf_counter()
    for _ in range(10):
        h = pol.net.encode(x)
        pol._units(raw, h)
        pol._market(raw, pol.net.market(h), None)
    print(f"numpy step (encoder + {int(raw['nunit'][0, 0])} units + market) {100 * (time.perf_counter() - t0):.1f} ms")


if __name__ == "__main__":
    main()
