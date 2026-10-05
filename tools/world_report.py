"""Pair two world_fork.py arms by world.

Reports wins (base -> candidate), strict flips (loss/tie -> win vs win -> loss/tie) with a two-sided sign test,
credible flips (the flip comes mostly from our own money: |d_me| > |d_opp|), the mean change in margin with its
standard error, and that change split into our side (d_me) and the opponent's (d_opp). A change that only makes the
opponent earn less ranks differently by margin than by wins, so read both.

  python tools/world_report.py results/d8/base.jsonl results/d8/cand.jsonl
"""
import json, math, sys

import numpy as np


def sign_test(up, dn):
    n = up + dn
    if n == 0:
        return 1.0
    k = min(up, dn)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def report(pairs, title):
    if not pairs:
        return
    d0 = np.array([b["me"] - b["op"] for b, c in pairs])
    d1 = np.array([c["me"] - c["op"] for b, c in pairs])
    dd = d1 - d0
    dme = np.array([c["me"] - b["me"] for b, c in pairs])
    dop = np.array([c["op"] - b["op"] for b, c in pairs])
    up, dn = (d0 <= 0) & (d1 > 0), (d0 > 0) & (d1 <= 0)
    cu, cd = up & (dme > np.abs(dop)), dn & (-dme > np.abs(dop))
    se = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else 0.0
    print(f"  {title:24s} n {len(pairs):4d}  wins {int((d0 > 0).sum()):3d} -> {int((d1 > 0).sum()):3d}  "
          f"flips +{int(up.sum())}/-{int(dn.sum())} (p {sign_test(int(up.sum()), int(dn.sum())):.3f})  "
          f"credible +{int(cu.sum())}/-{int(cd.sum())}  d_margin {dd.mean():+7,.0f} +- {se:5,.0f}  "
          f"d_me {dme.mean():+7,.0f}  d_opp {dop.mean():+7,.0f}")


def main():
    B, C = load(sys.argv[1]), load(sys.argv[2])
    P = [(B[i], C[i]) for i in B if i in C]
    print(f"base {sys.argv[1]}\ncand {sys.argv[2]}\nagent errors {sum(b['err'] for b, _ in P)} / {sum(c['err'] for _, c in P)}")
    report(P, "all")
    print("by recorded result")
    report([(b, c) for b, c in P if b["real"][0] <= b["real"][1]], "recorded loss / tie")
    report([(b, c) for b, c in P if b["real"][0] > b["real"][1]], "recorded win")
    print("by opponent")
    groups = {}
    for b, c in P:
        groups.setdefault(b["oteam"], []).append((b, c))
    for k in sorted(groups, key=lambda k: -len(groups[k])):
        report(groups[k], k[:24])


if __name__ == "__main__":
    main()
