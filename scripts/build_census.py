"""Replay JSON files -> census.jsonl (one row per episode) + episode / label caches + team vocabulary.

Stage 1 / 2 (all teams, your own seats excluded):
  python scripts/build_census.py --replays replays/ --out data/pretrain --since 2026-09-14 --exclude-teams "<YOUR TEAM>"
Stage 3 (one team's seats, all in one date bucket, vocabulary of the stage-1 model):
  python scripts/build_census.py --replays replays_target/ --out data/specialize --train-teams "<TARGET TEAM>" \
      --seat-day 2026-09-27 --vocab data/pretrain/vocab.json --drop-ids worlds/target.pkl

A replay's date is its parent directory name (YYYY-MM-DD), or --default-day. Episodes are kept if they have 720
steps, no ERROR / TIMEOUT / INVALID status and the given engine version. Seats that replay a fixed action tape
(the same team playing a byte-identical action stream in two or more episodes) are open-loop and teach nothing
about reacting to the state; their episodes are dropped.
"""
import argparse, collections, csv, datetime, glob, hashlib, json, os, pickle, re, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extract import extract_to  # noqa: E402
from labels import label_to  # noqa: E402

HEAD = 8192
RE_INFO = re.compile(rb'"info":\s*(\{.*?\})\s*,\s*"module_version"', re.S)
RE_REW = re.compile(rb'"rewards":\s*(\[[^\]]*\])')
RE_VER = re.compile(rb'"module_version":\s*"([^"]+)"')


def parse_header(path):
    """Episode id, seed, team names, rewards and engine version from the first 8 KB of a replay."""
    try:
        b = open(path, "rb").read(HEAD)
        info = json.loads(RE_INFO.search(b).group(1))
        return dict(path=str(path), id=info.get("EpisodeId"), seed=info.get("seed"), teams=info.get("TeamNames") or [],
                    rewards=json.loads(RE_REW.search(b).group(1)), ver=RE_VER.search(b).group(1).decode())
    except Exception as e:
        return dict(path=str(path), err=repr(e))


def audit(path):
    try:
        st = json.load(open(path)).get("steps", [])
        return len(st) == 720 and not any(a.get("status") in ("ERROR", "TIMEOUT", "INVALID") for s in st for a in s)
    except Exception:
        return False


def seat_hash(ep_path):
    a = np.load(ep_path)
    return [hashlib.sha1(a["uact"][:, s].tobytes() + a["mact"][:, s].tobytes() + a["nord"][:, s].tobytes()).hexdigest()
            for s in range(2)]


def replay_day(path, default):
    name = Path(path).parent.name
    try:
        datetime.date.fromisoformat(name)
        return name
    except ValueError:
        if default is None:
            raise SystemExit(f"cannot date {path}: parent directory is not YYYY-MM-DD and no --default-day given")
        return default


def read_ids(path):
    if path.endswith(".pkl"):
        return {int(w["id"]) for w in pickle.load(open(path, "rb"))}
    return {int(l.split(",")[0]) for l in open(path) if l.strip() and l.split(",")[0].strip().isdigit()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", nargs="+", required=True, help="directories searched recursively for *.json replays")
    ap.add_argument("--out", required=True, help="output directory for census.jsonl / vocab.json / report.json")
    ap.add_argument("--ep-dir", default=str(ROOT / "cache/ep"))
    ap.add_argument("--lab-dir", default=str(ROOT / "cache/lab"))
    ap.add_argument("--since", default="2000-01-01")
    ap.add_argument("--default-day", default=None)
    ap.add_argument("--version", default="1.32.7", help="kaggle-environments engine version to keep")
    ap.add_argument("--exclude-teams", nargs="*", default=[], help="seats of these teams are never trained")
    ap.add_argument("--train-teams", nargs="*", default=[], help="if given, only these teams' seats are trained")
    ap.add_argument("--seat-day", default=None, help="date bucket (YYYY-MM-DD) for every trained seat")
    ap.add_argument("--seat-days", default=None, help="CSV episode_id,seat,day overriding individual seats' dates")
    ap.add_argument("--drop-ids", nargs="*", default=[], help="episode ids to leave out (.pkl worlds or one id per line)")
    ap.add_argument("--keep-tapes", action="store_true", help="do not drop open-loop tape episodes")
    ap.add_argument("--vocab", default=None, help="reuse this vocabulary instead of building one")
    ap.add_argument("--min-seats", type=int, default=50, help="trainable seats a team needs to get its own embedding")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ep_dir, lab_dir = Path(a.ep_dir), Path(a.lab_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    rep = collections.Counter()

    files = sorted({p for d in a.replays for p in glob.glob(os.path.join(d, "**", "*.json"), recursive=True)})
    with Pool(a.procs) as pool:
        heads = pool.map(parse_header, files, chunksize=16)
    drop = set().union(*(read_ids(f) for f in a.drop_ids)) if a.drop_ids else set()
    rows, seen = [], set()
    for h in heads:
        if "err" in h or h["id"] is None:
            rep["unreadable"] += 1
            continue
        if h["id"] in seen:
            continue
        seen.add(h["id"])
        h["day"] = replay_day(h["path"], a.default_day)
        if h["day"] < a.since:
            continue
        if h["ver"] != a.version:
            rep["other_version"] += 1
            continue
        if h["id"] in drop:
            rep["dropped_by_id"] += 1
            continue
        rows.append(h)
    with Pool(a.procs) as pool:
        ok = pool.map(audit, [r["path"] for r in rows], chunksize=2)
    rep["failed_audit"] = sum(not g for g in ok)
    rows = [r for r, g in zip(rows, ok) if g]

    with Pool(a.procs) as pool:
        st = pool.map(extract_to, [(r["path"], str(ep_dir / f"{r['id']}.npz")) for r in rows], chunksize=2)
    bad = {r["id"] for r, s in zip(rows, st) if s.startswith("err")}
    rep["extract_errors"] = len(bad)
    rows = [r for r in rows if r["id"] not in bad]

    if not a.keep_tapes:
        with Pool(a.procs) as pool:
            hs = pool.map(seat_hash, [str(ep_dir / f"{r['id']}.npz") for r in rows], chunksize=16)
        cnt = collections.Counter((r["teams"][s], h[s]) for r, h in zip(rows, hs) for s in range(2))
        tape = {r["id"] for r, h in zip(rows, hs) if any(cnt[(r["teams"][s], h[s])] >= 2 for s in range(2))}
        rep["open_loop_tape"] = len(tape)
        rows = [r for r in rows if r["id"] not in tape]

    override = {}
    if a.seat_days:
        for rr in csv.reader(open(a.seat_days)):
            if rr and rr[0].strip().isdigit():
                override[(int(rr[0]), int(rr[1]))] = rr[2].strip()
    excl, only = set(a.exclude_teams), set(a.train_teams)
    for r in rows:
        mask, sd = [], []
        for s, t in enumerate(r["teams"]):
            m = t not in excl and (not only or t in only)
            d = override.get((r["id"], s), a.seat_day if (m and a.seat_day) else r["day"])
            mask.append(m)
            sd.append(d)
        r.update(mask=mask, seat_days=sd)
        del r["ver"]
    rows = [r for r in rows if any(r["mask"])]
    rows.sort(key=lambda r: (r["day"], r["id"]))

    if a.vocab:
        vocab = json.load(open(a.vocab))
    else:
        seats = collections.Counter(t for r in rows for s, t in enumerate(r["teams"]) if r["mask"][s])
        vocab = ["<other>"] + sorted(t for t, n in seats.items() if n >= a.min_seats)
    json.dump(vocab, open(out / "vocab.json", "w"), ensure_ascii=False)

    with Pool(a.procs) as pool:
        st = pool.map(label_to, [(str(ep_dir / f"{r['id']}.npz"), str(lab_dir / f"{r['id']}.npz")) for r in rows], chunksize=4)
    bad = {r["id"] for r, s in zip(rows, st) if s.startswith("err")}
    rep["label_errors"] = len(bad)
    rows = [r for r in rows if r["id"] not in bad]

    (out / "census.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    trained = collections.Counter(t for r in rows for s, t in enumerate(r["teams"]) if r["mask"][s])
    report = dict(files=len(files), episodes=len(rows), **rep, days=dict(sorted(collections.Counter(r["day"] for r in rows).items())),
                  vocab=len(vocab), trained_seats_by_team=dict(trained.most_common()))
    json.dump(report, open(out / "report.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
