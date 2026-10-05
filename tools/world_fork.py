"""Fork evaluation on recorded games.

For every world, both sides replay their recorded actions up to the start of day --day; from then on the arm's
agent plays our seat and the opponent keeps following its recording. Before the fork the agent still sees every
observation (its actions are discarded) so its internal state is the one it would have had. Shops are pinned to
the recorded ones. Each arm runs every world; tools/world_report.py pairs arms by world.

  python tools/world_fork.py cand=agents/new base=agents/old --worlds worlds/mine.pkl --day 8 --out results/d8

Needs kagsim (https://github.com/destbreso/kaggriculture-cppsim) and kaggle-environments.
"""
import argparse, inspect, json, os, pickle, time
from multiprocessing import get_context

_CFG = None


def load_agent(main_py):
    """Load main.py the way the Kaggle runner does (compile + exec, last callable is the agent)."""
    from kaggle_environments.agent import get_last_callable
    main_py = os.path.abspath(main_py)
    return get_last_callable(open(main_py).read(), path=main_py)


def run(job):
    global _CFG
    import kagsim
    if _CFG is None:
        from kaggle_environments import make
        _CFG = make("kaggriculture", configuration={"episodeSteps": 720}).configuration
    name, adir, w, day = job
    fn = load_agent(os.path.join(adir, "main.py"))
    two_args = len(inspect.signature(fn).parameters) >= 2
    g = kagsim.Game(w["seed"], 720, w["shops"])
    T = 24 * day
    worst = 0.0
    for t in range(719):
        o = g.observe(w["me"])
        t0 = time.perf_counter()
        a = fn(o, _CFG) if two_args else fn(o)
        worst = max(worst, time.perf_counter() - t0)
        acts = [None, None]
        acts[w["me"]] = w["mytape"][t] if t < T else a
        acts[w["op"]] = w["tape"][t]
        g.step(acts[0], acts[1])
    stats = fn.__globals__.get("AGENT_STATS", {})
    return dict(arm=name, id=w["id"], me=float(g.reward(w["me"])), op=float(g.reward(w["op"])),
                err=int(stats.get("errors", 0)), max_call=round(worst, 3), oteam=w["oteam"],
                real=[w["rewards"][w["me"]], w["rewards"][w["op"]]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="NAME=AGENT_DIR (a directory with main.py)")
    ap.add_argument("--worlds", nargs="+", required=True)
    ap.add_argument("--day", type=int, default=8)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W, seen = [], set()
    for f in a.worlds:
        for w in pickle.load(open(f, "rb")):
            if w["id"] not in seen:
                seen.add(w["id"])
                W.append(w)
    arms = [x.split("=", 1) for x in a.arms]
    jobs = [(n, os.path.abspath(d), w, a.day) for n, d in arms for w in W]
    t0 = time.time()
    with get_context("spawn").Pool(a.procs, maxtasksperchild=8) as pool:
        res = pool.map(run, jobs, chunksize=1)
    os.makedirs(a.out, exist_ok=True)
    for n, d in arms:
        rows = [r for r in res if r["arm"] == n]
        with open(os.path.join(a.out, f"{n}.jsonl"), "w") as fo:
            fo.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"{n}: {len(rows)} worlds, won {sum(r['me'] > r['op'] for r in rows)}, agent errors {sum(r['err'] for r in rows)}, "
              f"slowest call {max(r['max_call'] for r in rows):.2f}s ({d})", flush=True)
    print(f"day-{a.day} forks: {len(W)} worlds x {len(arms)} arms in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
