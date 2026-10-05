"""Episode npz (extract.py) -> BC label npz.

Unit labels, per unit and step:
  u_op / u_tile  the unit's "task": its first non-move action between now and the end of the day, and the tile it
                 stands on when doing it. If it only walks for the rest of the day, the task is PASS and the target is
                 where it stops.
  u_item / u_qty item and effective quantity of PICKUP / PLACE / PLANT tasks.
The raw per-step action (including moves) is read straight from the episode arrays at training time.

Market labels are per-product quantities. SELL is capped at what the shed actually holds after this step's unit
actions. BUY_ANIMAL counts animals that actually arrived (failed orders would otherwise teach the model noise).
"""
import os

import numpy as np

from extract import UOP, MOP, MAXO
from common import MAXU, TASK_OPS

MOVES = (1, 2, 3, 4)
RAW2TASK = np.full(18, -1, np.int8)
for k, v in UOP.items():
    if k in TASK_OPS:
        RAW2TASK[v] = TASK_OPS.index(k)
SHED_TILES = {(4, 4), (5, 4), (4, 5), (5, 5)}
STRUCT = {9: 4, 10: 5, 11: 5}  # animal item id -> building kind (COOP = 4, PASTURE = 5)
DXY = {1: (0, -1), 2: (0, 1), 3: (1, 0), 4: (-1, 0)}


def unit_phase_shed(shed, inv_t, upos_t, uact_t, tile_t, n):
    """Apply this step's unit actions to the shed in engine order; returns (shed, bags) as the market sees them."""
    shed = shed.astype(np.int32).copy()
    inv = inv_t.astype(np.int32).copy()
    tot = int(shed.sum())
    for i in range(n):
        op, item, q = int(uact_t[i, 0]), int(uact_t[i, 1]) - 1, int(uact_t[i, 2])
        x, y = int(upos_t[i, 0]), int(upos_t[i, 1])
        at_shed = (x, y) in SHED_TILES
        if op == UOP["DROP"] and at_shed:
            for it in range(12):
                if inv[i, it] > 0:
                    take = min(inv[i, it], max(0, 100 - tot))
                    shed[it] += take
                    tot += take
                    inv[i, it] = 0
        elif op == UOP["PICKUP"] and at_shed and 0 <= item < 12 and q > 0:
            k = min(q, shed[item])
            shed[item] -= k
            tot -= k
            inv[i, item] += k
        elif op == UOP["PLACE"] and 0 <= item < 12:
            tl = tile_t[y * 10 + x]
            if item in STRUCT and tl[0] == STRUCT[item] and tl[1] == 0:
                if inv[i, item] > 0:
                    inv[i, item] -= 1
                continue
            if at_shed and q > 0:
                k = min(q, inv[i, item], max(0, 100 - tot))
                if k > 0:
                    inv[i, item] -= k
                    shed[item] += k
                    tot += k
    return shed, inv


def make_labels(ep):
    T = 720
    uact, upos, nunit = ep["uact"], ep["upos"], ep["nunit"]
    u_op = np.full((T, 2, MAXU), -1, np.int8)
    u_tile = np.zeros((T, 2, MAXU), np.int8)
    u_item = np.zeros((T, 2, MAXU), np.int8)      # 0 = none, 1..12 = ITEMS
    u_qty = np.zeros((T, 2, MAXU), np.int16)
    m_sell = np.zeros((T, 2, 9), np.int16)
    m_seed = np.zeros((T, 2, 5), np.int32)
    m_buyp = np.zeros((T, 2, 2), np.int32)
    m_anim = np.zeros((T, 2, 3), np.int32)
    m_hire = np.zeros((T, 2), np.int8)
    m_land = np.zeros((T, 2), np.int8)
    for s in range(2):
        nu = nunit[:, s].astype(int)
        for t in range(T - 1):
            day_end = min(t // 24 * 24 + 23, T - 2)
            for i in range(nu[t]):
                t2 = t
                while t2 <= day_end and uact[t2, s, i, 0] in MOVES and i < nu[t2]:
                    t2 += 1
                if t2 <= day_end and i < nu[t2]:
                    op = int(uact[t2, s, i, 0])
                    x, y = int(upos[t2, s, i, 0]), int(upos[t2, s, i, 1])
                    u_op[t, s, i] = RAW2TASK[op]
                    it = int(uact[t2, s, i, 1])
                    q = int(uact[t2, s, i, 2])
                    if op == UOP["PICKUP"] and 1 <= it <= 12:
                        q = min(q, int(ep["shed"][t2, s, it - 1]))
                    elif op == UOP["PLACE"] and 1 <= it <= 12:
                        q = min(q, int(ep["inv"][t2, s, i, it - 1]))
                    u_item[t, s, i] = it if op in (UOP["PICKUP"], UOP["PLACE"], UOP["PLANT"]) else 0
                    u_qty[t, s, i] = max(0, q) if op in (UOP["PICKUP"], UOP["PLACE"]) else 0
                else:
                    x, y = int(upos[t, s, i, 0]), int(upos[t, s, i, 1])
                    for tt in range(t, t2):
                        dx, dy = DXY[int(uact[tt, s, i, 0])]
                        if 0 <= x + dx < 10 and 0 <= y + dy < 10:
                            x, y = x + dx, y + dy
                    u_op[t, s, i] = 0
                u_tile[t, s, i] = y * 10 + x
            pj, pinv = unit_phase_shed(ep["shed"][t, s], ep["inv"][t, s], upos[t, s], uact[t, s], ep["tile"][t, s], nu[t])
            sell_raw = np.zeros(9, np.int32)
            anim_raw = np.zeros(3, np.int32)
            for k in range(min(int(ep["nord"][t, s]), MAXO)):
                op, it, q = (int(v) for v in ep["mact"][t, s, k])
                it -= 1
                if op == MOP["SELL"] and 0 <= it < 9:
                    sell_raw[it] += max(0, q)
                elif op == MOP["BUY_SEED"] and 0 <= it < 5:
                    m_seed[t, s, it] += max(0, q)
                elif op == MOP["BUY_PRODUCT"] and it in (0, 8):
                    m_buyp[t, s, 0 if it == 0 else 1] += max(0, q)
                elif op == MOP["BUY_ANIMAL"] and 9 <= it < 12:
                    anim_raw[it - 9] += max(0, q)
                elif op == MOP["HIRE"]:
                    m_hire[t, s] += 1
                elif op == MOP["BUY_LAND"]:
                    m_land[t, s] = 1
            m_sell[t, s] = np.minimum(sell_raw, pj[:9])
            for k in range(3):
                if anim_raw[k] > 0:
                    got = int(ep["shed"][t + 1, s, 9 + k]) - int(pj[9 + k])
                    if (t + 1) % 24 == 0:          # end of day: bags (incl. unplaced animals) are emptied into the shed
                        got -= int(pinv[:nu[t], 9 + k].sum())
                    m_anim[t, s, k] = min(max(got, 0), anim_raw[k])
    return dict(u_op=u_op, u_tile=u_tile, u_item=u_item, u_qty=u_qty, m_sell=m_sell, m_seed=m_seed, m_buyp=m_buyp,
                m_anim=m_anim, m_hire=m_hire, m_land=m_land)


def label_to(args):
    """(episode npz, label npz) -> status string. Skips existing outputs; writes atomically."""
    src, dst = args
    if os.path.exists(dst):
        return "skip"
    try:
        lab = make_labels(dict(np.load(src)))
        tmp = dst + ".tmp.npz"
        np.savez_compressed(tmp, **lab)
        os.replace(tmp, dst)
        return "ok"
    except Exception as e:
        return f"err {os.path.basename(src)} {e!r:.200}"
