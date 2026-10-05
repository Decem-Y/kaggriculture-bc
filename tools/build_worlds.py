"""Recorded ladder games -> "worlds" for fork evaluation (tools/world_fork.py).

A world is a recorded episode: seed, the shops that were unlocked, and both sides' action tapes. Only episodes
the simulator reproduces to the dollar from the tapes are kept.

  # games your team played: the candidate takes over your seat
  python tools/build_worlds.py --replays replays/mine --team "<YOUR TEAM>" --out worlds/mine.pkl
  # games a target team played against anyone else: the candidate takes over the seat facing it
  python tools/build_worlds.py --replays replays/target --opponent "<TARGET TEAM>" --out worlds/target.pkl

Needs kagsim (https://github.com/destbreso/kaggriculture-cppsim).
"""
import argparse, glob, json, os, pickle
from collections import Counter
from multiprocessing import Pool

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def load_world(args):
    path, team, opponent = args
    try:
        ep = json.load(open(path))
    except Exception:
        return None, "unreadable"
    st = ep.get("steps", [])
    teams = ep.get("info", {}).get("TeamNames") or []
    if len(st) != 720 or len(teams) != 2:
        return None, "incomplete"
    if team:
        if teams.count(team) != 1:
            return None, "team_not_in_game"
        me = teams.index(team)
    else:
        if teams.count(opponent) != 1:
            return None, "opponent_not_in_game"
        me = 1 - teams.index(opponent)
    op = 1 - me

    def tape(s):
        return [st[t + 1][s].get("action") if isinstance(st[t + 1][s].get("action"), dict) else PASS for t in range(719)]
    w = dict(id=int(ep["info"]["EpisodeId"]), seed=int(ep["info"]["seed"]),
             shops=list(st[-1][0]["observation"]["town"]["unlocked_shops"]), me=me, op=op, mytape=tape(me), tape=tape(op),
             rewards=[float(x) for x in ep["rewards"]], myteam=teams[me], oteam=teams[op], path=path)
    if replay(w) != w["rewards"]:
        return None, "not_reproduced"
    return w, "ok"


def replay(w):
    import kagsim
    g = kagsim.Game(w["seed"], 720, w["shops"])
    for t in range(719):
        a = [None, None]
        a[w["me"]], a[w["op"]] = w["mytape"][t], w["tape"][t]
        g.step(a[0], a[1])
    return [float(g.reward(0)), float(g.reward(1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", nargs="+", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--team", help="take over this team's seat")
    g.add_argument("--opponent", help="take over the seat facing this team")
    ap.add_argument("--exclude", nargs="*", default=[], help="episode ids already in other world files (.pkl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    skip = {int(w["id"]) for f in a.exclude for w in pickle.load(open(f, "rb"))}
    files = sorted({p for d in a.replays for p in glob.glob(os.path.join(d, "**", "*.json"), recursive=True)})
    with Pool(a.procs) as pool:
        res = pool.map(load_world, [(p, a.team, a.opponent) for p in files], chunksize=4)
    stat = Counter(s for _, s in res)
    W, seen = [], set(skip)
    for w, _ in res:
        if w is not None and w["id"] not in seen:
            seen.add(w["id"])
            W.append(w)
    W.sort(key=lambda w: w["id"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pickle.dump(W, open(a.out, "wb"))
    won = sum(w["rewards"][w["me"]] > w["rewards"][w["op"]] for w in W)
    print(f"{len(W)} worlds ({won} won in the recording) -> {a.out}; {dict(stat)}")
    print("opponents:", dict(Counter(w["oteam"] for w in W).most_common()))


if __name__ == "__main__":
    main()
