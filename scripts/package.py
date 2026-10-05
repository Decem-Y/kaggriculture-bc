"""Pack an agent directory (main.py, weights.npz, vocab.json) into a submission tar.gz and check it the way the
Kaggle runner will use it.

  (a) the archive is deterministic, round-trips byte for byte and is under 100 MiB;
  (b) main.py is loaded from the extracted copy with the runner's loader, from an unrelated working directory;
  (c) full games on the official engine, both seats: no ERROR / TIMEOUT, no exception caught inside the agent,
      slowest single call under --max-call seconds;
  (d) a second game in the same process equals the same game in a fresh process (no state leaks across episodes),
      played against the built-in deterministic "starter" agent;
  (e) with --parity and kagsim installed, kagsim reproduces every official game to the dollar.

  python scripts/package.py agents/final --out artifacts/final.tar.gz [--opponents random path/to/other/main.py]
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse, gzip, hashlib, io, json, shutil, sys, tarfile, tempfile, time
from multiprocessing import Pool
from pathlib import Path

FILES = ("main.py", "weights.npz", "vocab.json")
_CFG = None


def load(path):
    from kaggle_environments.agent import get_last_callable
    if not path.endswith(".py"):
        return path                       # built-in agent name, e.g. "random"
    path = os.path.abspath(path)
    return get_last_callable(open(path).read(), path=path)


def timed(fn, box):
    def call(obs, cfg=None):
        t0 = time.perf_counter()
        a = fn(obs, cfg)
        box.append(time.perf_counter() - t0)
        return a
    return call


def official(me_path, opp, seed, seat):
    from kaggle_environments import make
    fn = load(me_path)
    calls = []
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    agents = [timed(fn, calls), load(opp)]
    env.run(agents if seat == 0 else agents[::-1])
    bad = sorted({s.status for st in env.steps for s in st if s.status in ("ERROR", "TIMEOUT", "INVALID")})
    rewards = [float(s.reward) for s in env.steps[-1]]
    errors = fn.__globals__.get("AGENT_STATS", {}).get("errors", 0)
    return dict(rewards=rewards, bad=bad, errors=errors, max_call=max(calls), steps=env.steps)


def game(job):
    me_path, opp, seed, seat, cwd, parity = job
    os.chdir(cwd)
    r = official(me_path, opp, seed, seat)
    out = dict(opponent=opp, seed=seed, seat=seat, rewards=r["rewards"], bad=r["bad"], errors=r["errors"], max_call=r["max_call"])
    if parity:
        import kagsim
        acts = [[st[s].action for st in r["steps"][1:]] for s in range(2)]
        g = kagsim.Game(seed)
        for t in range(719):
            g.step(acts[0][t], acts[1][t])
        out["kagsim"] = [float(g.reward(0)), float(g.reward(1))]
    return out


def twice(job):
    """Two games in one process; the second must equal a fresh process playing it."""
    me_path, opp, seeds, cwd = job
    os.chdir(cwd)
    from kaggle_environments import make
    fn = load(me_path)

    def play(f, seed):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        env.run([f, load(opp)])
        return [float(s.reward) for s in env.steps[-1]]
    play(fn, seeds[0])
    return play(fn, seeds[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opponents", nargs="*", default=["starter", "random"], help="built-in names or main.py paths")
    ap.add_argument("--seeds", nargs="*", type=int, default=[210007, 210008])
    ap.add_argument("--max-call", type=float, default=0.85)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    src = Path(a.agent_dir)
    files = {n: (src / n).read_bytes() for n in FILES}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tf:
        for n, data in sorted(files.items()):
            info = tarfile.TarInfo(n)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tf.addfile(info, io.BytesIO(data))
    size = out.stat().st_size
    assert size < 100 * 1024 * 1024, f"archive is {size:,} B, over the 100 MiB limit"

    tmp = Path(tempfile.mkdtemp(prefix="pkgcheck-"))
    cwd = tempfile.mkdtemp(prefix="pkgcwd-")
    try:
        with tarfile.open(out) as tf:
            tf.extractall(tmp, filter="data")
        assert {p.name: p.read_bytes() for p in tmp.iterdir()} == files, "archive does not round-trip"
        main_py = str(tmp / "main.py")
        opps = [o if not o.endswith(".py") else os.path.abspath(o) for o in a.opponents]
        jobs = [(main_py, o, s, seat, cwd, a.parity) for o in opps for s in a.seeds for seat in (0, 1)]
        with Pool(a.procs) as pool:
            res = pool.map(game, jobs, chunksize=1)
            second = pool.apply(twice, ((main_py, "starter", a.seeds, cwd),))
            fresh = pool.apply(game, ((main_py, "starter", a.seeds[1], 0, cwd, False),))["rewards"]
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(cwd)

    ok = True
    for r in res:
        good = not r["bad"] and r["errors"] == 0 and r["max_call"] < a.max_call
        if a.parity:
            good &= r["kagsim"] == r["rewards"]
        ok &= good
        print(f"{Path(r['opponent']).parent.name or r['opponent']:<20} seed {r['seed']} seat {r['seat']}  {r['rewards']}  "
              f"slowest {r['max_call']:.3f}s  {'ok' if good else 'FAIL ' + json.dumps({k: r[k] for k in ('bad', 'errors')})}"
              + (f"  kagsim {r['kagsim']}" if a.parity else ""))
    print(f"second game in one process {second} vs fresh process {fresh}: {'same' if second == fresh else 'DIFFERENT'}")
    ok &= second == fresh
    print(f"{out}  {size:,} B  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
