"""BC policy: observation -> action, numpy only.

Units are decoded one at a time. Unit i scores (task, target tile) jointly given the choices of the units before it,
then picks a raw action (moves included) conditioned on task, target and the offset to it. With topk > 0 the raw
action is marginalised over the top-k (task, tile) plans. Raw actions are masked to those that are valid on the
unit's current tile; items and quantities are clipped against a running simulation of this step's state, which
later units and the market layer read.

Market orders: per-product quantity bins, turned into orders through expected-quantity accumulators. Order is
SELL -> HIRE -> LAND -> ANIMAL -> SEED -> BUY_PRODUCT.

Two rules on top of the model, both only recover goods the engine would destroy:
- overflow sell: on the last step of each day, sell from the shed whatever the end-of-day bag dump would push
  past the 100-item cap;
- final-day return: day 29 has no end-of-day dump, so units carrying goods walk back to the shed and DROP in time.
"""
import json, os

import numpy as np

from common import obs_to_raw, featurize, QTY_BINS, BUY_BINS, TASK_OPS, UNIT_OPS, ITEMS, ITEM_ID, CROPS, ANIMALS, PRODUCTS
from npmodel import NPNet

OP = {k: i for i, k in enumerate(TASK_OPS)}
NOP = len(TASK_OPS)
R = {k: i for i, k in enumerate(UNIT_OPS)}
NRAW = len(UNIT_OPS)
SHED = np.zeros(100, bool)
SHED[[44, 45, 54, 55]] = True
SHED_XY = ((4, 4), (5, 4), (4, 5), (5, 5))
FIRST_YIELD = np.array([0, 2, 2, 8, 10, 10], np.int32)      # indexed by crop id + 1
STRUCT_OF = {9: 4, 10: 5, 11: 5}                             # animal item -> building kind


def _lsm(x, axis=-1):
    m = x.max(axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis, keepdims=True))


class BCPolicy:
    def __init__(self, wdir, team=None, won=1, date=0, mask=True, topk=0, rules=True):
        self.mask = mask            # False: no legality mask (only for comparing against the torch decoder)
        self.topk = topk
        self.rules = rules
        self.net = NPNet(os.path.join(wdir, "weights.npz"))
        vocab = json.load(open(os.path.join(wdir, "vocab.json")))
        self.team = vocab.index(team) if team in vocab else 0
        self.won = won
        self.date = date
        self.last_step = -1
        self._reset_acc()

    def _reset_acc(self):
        # Expected-quantity accumulators. Per-step argmax says 0 forever when the model is unsure about timing.
        self.acc = {"sell": np.zeros(9), "land": np.zeros(1), "anim": np.zeros(3), "seed": np.zeros(5), "buyp": np.zeros(2)}

    def act(self, obs):
        step = int(obs.get("step", 0) or 0)
        if step <= self.last_step:          # a new episode in the same process
            self._reset_acc()
        self.last_step = step
        raw = obs_to_raw(obs)
        x = featurize(raw, np.array([self.team]), np.array([self.won]), np.array([self.date]))
        h = self.net.encode(x)
        mk = self.net.market(h)
        units = self._units(raw, h)
        market = self._market(raw, mk, obs)
        act = {"farmer": units[0], "hands": units[1:], "market": market}
        if self.rules:
            act = self._overflow_sell(act, obs, step)
            act = self._final_return(act, obs, step)
        return act

    # ---- units ----
    def _units(self, raw, h):
        tl = raw["tiles"][0, 0].astype(np.int32).copy()
        kind, sub, age, yld, wet = tl[:, 0], tl[:, 1], tl[:, 2], tl[:, 3], tl[:, 4]
        fert, cared = tl[:, 6], tl[:, 8]
        shed = raw["shed"][0].astype(np.int32).copy()
        seeds = raw["seeds"][0].astype(np.int32).copy()
        n = int(raw["nunit"][0, 0])
        pos = raw["upos"][0, 0, :n].astype(np.int32)
        inv = raw["inv"][0, :n].astype(np.int32).copy()
        shed0, kind0, sub0 = shed.copy(), kind.copy(), sub.copy()   # start-of-step state
        ctx = self.net.unit_ctx(h)
        out = []
        for i in range(n):
            px, py = int(pos[i, 0]), int(pos[i, 1])
            here = py * 10 + px
            zf = self.net.unit_query(ctx, i, here)
            lop, lt = self.net.unit_scores(ctx, zf)
            sc = (lop[:, None] + lt).reshape(-1)
            legal = (self._legal(i, px, py, here, kind, sub, yld, wet, age, fert, cared, shed, shed0, kind0, sub0, seeds, inv)
                     if self.mask else np.ones(NRAW, bool))
            if self.topk:
                idx = np.argsort(-sc, kind="stable")[:self.topk]
                lp = sc[idx]
                lr = np.stack([_lsm(self.net.raw_logits(ctx, zf, int(j) // 100, int(j) % 100, here)) for j in idx])
                mx = (lp[:, None] + lr).max(0)
                marg = mx + np.log(np.exp(lp[:, None] + lr - mx).sum(0))
                marg[~legal] = -np.inf
                rw = int(np.argmax(marg))
                kk = int(np.argmax(lp + lr[:, rw]))
                op, tgt = int(idx[kk]) // 100, int(idx[kk]) % 100
            else:
                op, tgt = divmod(int(np.argmax(sc)), 100)
                rl = self.net.raw_logits(ctx, zf, op, tgt, here).copy()
                rl[~legal] = -np.inf
                rw = int(np.argmax(rl))
            il, ql = self.net.arg_logits(ctx, zf, rw, tgt)
            act, item = self._emit(i, rw, il, ql, here, shed, shed0, kind0, sub0, seeds, inv, kind, sub, yld, wet, fert, cared)
            self.net.push_memory(ctx, i, op, tgt, rw, item)
            out.append(act)
        self._proj_shed = shed.copy()          # shed after the unit phase = what the market sees
        self._proj_inv = inv.copy()            # bags after the unit phase
        return out

    @staticmethod
    def _legal(i, px, py, here, kind, sub, yld, wet, age, fert, cared, shed, shed0, kind0, sub0, seeds, inv):
        """Which of the 18 raw actions do something on this tile. The rest are silent no-ops in the engine, or worse:
        DROP into a full shed destroys the bag, and over-planting voids every PLANT of that crop."""
        ok = np.zeros(NRAW, bool)
        ok[0] = True
        ok[1], ok[2], ok[3], ok[4] = py > 0, py < 9, px < 9, px > 0
        k, s = kind[here], sub[here]
        plant, empty, animal = k == 3, k == 0, (k >= 4) & (s > 0)
        iv = inv[i]
        room = 100 - int(shed.sum())
        at_shed = bool(SHED[here])
        ok[R["PICKUP"]] = at_shed and shed0.sum() > 0
        ok[R["PLANT"]] = empty and seeds.sum() > 0
        ok[R["WATER"]] = plant and wet[here] == 0
        ok[R["HARVEST"]] = (plant and yld[here] > 0 and age[here] >= FIRST_YIELD[min(max(s, 0), 5)]) or (animal and yld[here] > 0)
        ok[R["FERTILIZE"]] = plant and iv[8] > 0
        ok[R["BUILD_COOP"]] = empty
        ok[R["BUILD_PASTURE"]] = empty
        ok[R["DIG"]] = k not in (0, 1) and not animal
        can_place_animal = any(iv[a] > 0 and kind0[here] == STRUCT_OF[a] and sub0[here] == 0 for a in (9, 10, 11))
        ok[R["PLACE"]] = can_place_animal or (at_shed and iv[:9].sum() > 0 and room > 0)
        ok[R["FEED"]] = animal and wet[here] == 0 and iv[0] > 0
        ok[R["COLLECT_FERTILIZER"]] = animal and fert[here] > 0
        ok[R["CARE"]] = animal and cared[here] == 0
        ok[R["DROP"]] = at_shed and 0 < iv.sum() <= room
        return ok

    def _emit(self, i, rw, il, ql, t, shed, shed0, kind0, sub0, seeds, inv, kind, sub, yld, wet, fert, cared):
        """Raw action -> engine action; updates the simulated step state. Returns (action, item id for the memory)."""
        name = UNIT_OPS[rw]
        iv = inv[i]
        il = il.copy()
        il[0] = -np.inf
        if name in ("PASS", "NORTH", "SOUTH", "EAST", "WEST"):
            return [name], 0
        if name == "PLANT":
            ok = np.zeros(13, bool)
            ok[1:6] = seeds > 0
            if not ok.any():
                return ["PASS"], 0
            il[~ok] = -np.inf
            c = int(np.argmax(il)) - 1
            seeds[c] -= 1
            kind[t], sub[t], yld[t], wet[t] = 3, c + 1, (0 if c in (2, 3) else 1), 0
            return ["PLANT", CROPS[c]], c + 1
        if name == "PICKUP":
            ok = np.zeros(13, bool)
            ok[1:] = shed0 > 0
            if not ok.any():
                return ["PASS"], 0
            il[~ok] = -np.inf
            it = int(np.argmax(il)) - 1
            q = int(min(max(1, QTY_BINS[int(np.argmax(ql))]), shed0[it]))
            got = min(q, int(shed[it]))
            shed[it] -= got
            iv[it] += got
            return ["PICKUP", ITEMS[it], q], it + 1
        if name == "PLACE":
            ok_a = [a for a in (9, 10, 11) if iv[a] > 0 and kind0[t] == STRUCT_OF[a] and sub0[t] == 0]
            if ok_a:
                a = max(ok_a, key=lambda b: il[b + 1])
                if sub[t] == 0:
                    iv[a] -= 1
                    sub[t] = a - 8
                    wet[t] = 0
                return ["PLACE", ITEMS[a], 1], a + 1
            ok = np.zeros(13, bool)
            ok[1:10] = iv[:9] > 0
            if not ok.any():
                return ["PASS"], 0
            il[~ok] = -np.inf
            it = int(np.argmax(il)) - 1
            room = 100 - int(shed.sum())
            q = int(min(max(1, QTY_BINS[int(np.argmax(ql))]), iv[it], max(room, 0)))
            iv[it] -= q
            shed[it] += q
            return ["PLACE", ITEMS[it], q], it + 1
        if name == "DROP":
            shed += iv
            iv[:] = 0
        elif name == "WATER":
            wet[t] = 1
        elif name == "HARVEST":
            if kind[t] == 3:
                c = sub[t] - 1
                iv[c] += yld[t]
                yld[t] = 0
                if c in (0, 1, 4):
                    kind[t], sub[t] = 0, 0
            elif sub[t] > 0:
                iv[[5, 6, 7][sub[t] - 1]] += yld[t]
                yld[t] = 0
        elif name == "FERTILIZE":
            iv[8] -= 1
            fert[t] = 3
        elif name == "FEED":
            iv[0] -= 1
            wet[t] = 1
        elif name == "CARE":
            cared[t] = 1
        elif name == "COLLECT_FERTILIZER":
            fert[t] = 0
            iv[8] += 1
        elif name == "DIG":
            kind[t], sub[t] = 0, 0
        elif name == "BUILD_COOP":
            kind[t] = 4
        elif name == "BUILD_PASTURE":
            kind[t] = 5
        return [name], 0

    # ---- market ----
    @staticmethod
    def _eq(logits, bins):
        pr = np.exp(logits - logits.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        return pr @ bins.astype(np.float64), bins[np.argmax(logits, -1)]

    def _take(self, key, idx, eq, am, cap=None):
        """q = max(argmax, floor(accumulated E[q])): sharp decisions follow the argmax, rare events whose timing the
        model spreads over many steps are realised by the accumulator."""
        acc = self.acc[key]
        acc[idx] += eq
        if cap is not None:
            acc[idx] = min(acc[idx], cap)
        q = max(int(am), int(acc[idx]))
        if cap is not None:
            q = min(q, int(cap))
        acc[idx] = max(0.0, acc[idx] - q)
        return q

    def _market(self, raw, mk, obs):
        orders = []
        step = int(raw["step"][0])
        price = raw["mkt_price"][0].astype(np.int64)
        shed = self._proj_shed
        final = step >= 718                      # last actionable step: unsold stock is worth nothing
        sells = []
        eq, am = self._eq(mk["sell"][0], QTY_BINS)
        for p in range(9):
            if shed[p] <= 0:
                self.acc["sell"][p] = 0.0
                continue
            q = int(shed[p]) if final else self._take("sell", p, eq[p], am[p], cap=float(shed[p]))
            if q > 0:
                sells.append((q * price[p], ["SELL", PRODUCTS[p], q]))
        sells.sort(key=lambda z: -z[0])
        orders += [s[1] for s in sells]
        k = int(np.argmax(mk["hire"][0]))
        orders += [["HIRE"]] * k
        lp = mk["land"][0]
        pl = float(np.exp(lp[1] - lp.max()) / np.exp(lp - lp.max()).sum())
        if int(raw["quads"][0, 0]) < 4 and self._take("land", 0, pl, int(np.argmax(lp)), cap=1.0) > 0:
            orders.append(["BUY_LAND"])
        eq, am = self._eq(mk["anim"][0], np.arange(mk["anim"].shape[-1]))
        for a in range(3):
            q = self._take("anim", a, eq[a], am[a], cap=8.0)
            if q > 0:
                orders += [["BUY_ANIMAL", ANIMALS[a], 1] for _ in range(q)]
        eq, am = self._eq(mk["seed"][0], BUY_BINS)
        for c in range(5):
            q = self._take("seed", c, eq[c], am[c], cap=200.0)
            if q > 0:
                orders.append(["BUY_SEED", CROPS[c], q])
        eq, am = self._eq(mk["buyp"][0], BUY_BINS)
        for j, it in enumerate(["WHEAT", "FERTILIZER"]):
            q = self._take("buyp", j, eq[j], am[j], cap=200.0)
            if q > 0:
                orders.append(["BUY_PRODUCT", it, q])
        if len(orders) > 10:                   # only 10 orders execute: keep the HIRE orders, drop the smallest sells first
            _s = [o for o in orders if o[0] == "SELL"]
            _h = [o for o in orders if o[0] == "HIRE"]
            _r = [o for o in orders if o[0] not in ("SELL", "HIRE")]
            orders = _s[:max(0, min(len(_s), 10 - len(_h)))] + _h + _r
        return orders[:10]

    # ---- rules ----
    def _overflow_sell(self, act, obs, step):
        """Last step of each day (except 718): the end-of-day dump empties all bags into the shed and the engine drops
        whatever exceeds 100 items, in order farmer -> hands, bag insertion order within a bag. Work out which goods
        would be dropped and sell the same goods from the shed now; if the shed lacks them, sell the cheapest
        products instead (never one being bought this step). Merges into existing SELL orders, otherwise appends
        while fewer than 10 orders are queued."""
        inv = getattr(self, "_proj_inv", None)
        if step % 24 != 23 or step >= 718 or inv is None:
            return act
        orders = act["market"]
        shed = np.asarray(self._proj_shed, np.int64)
        inv = np.asarray(inv, np.int64)
        pack = int(inv.sum())
        if pack <= 0:
            return act
        price = np.array([int(obs["market"]["prices"][p]) for p in PRODUCTS], np.int64)
        sold = np.zeros(12, np.int64)
        buy, bought = 0, set()
        for o in orders:
            if o[0] == "SELL" and len(o) > 2:
                sold[ITEM_ID[o[1]]] += int(o[2])
            elif o[0] == "BUY_PRODUCT" and len(o) > 2:
                buy += int(o[2])
                bought.add(ITEM_ID[o[1]])
            elif o[0] == "BUY_ANIMAL":
                buy += int(o[2]) if len(o) > 2 else 1
        sold = np.minimum(sold, shed)
        x = min(pack, int(shed.sum() - sold.sum()) + buy + pack - 100)
        if x <= 0:
            return act
        bags = (obs.get("private") or {}).get("inventories") or []
        seq = []
        for u in range(inv.shape[0]):
            keys = [ITEM_ID[k] for k in (bags[u] if u < len(bags) else {}) if k in ITEM_ID]
            keys += [j for j in range(12) if j not in keys]
            for j in keys:
                seq += [j] * int(inv[u, j])
        tail = np.bincount(np.array(seq[-x:], np.int64), minlength=12)
        avail = shed - sold
        add = np.zeros(9, np.int64)
        for p in range(9):
            add[p] = min(int(tail[p]), int(avail[p]))
        rest = x - int(add.sum())
        for p in np.argsort(price, kind="stable"):
            if rest <= 0:
                break
            if int(p) in bought:
                continue
            q = min(rest, int(avail[p] - add[p]))
            if q > 0:
                add[p] += q
                rest -= q
        last = max([i for i, o in enumerate(orders) if o[0] == "SELL"], default=-1)
        for p in np.argsort(-price * add, kind="stable"):
            p = int(p)
            if add[p] <= 0:
                continue
            hit = [o for o in orders if o[0] == "SELL" and o[1] == PRODUCTS[p]]
            if hit:
                hit[0][2] = int(hit[0][2]) + int(add[p])
            elif len(orders) < 10:
                last += 1
                orders.insert(last, ["SELL", PRODUCTS[p], int(add[p])])
        return act

    def _final_return(self, act, obs, step):
        """Day 29 has no end-of-day dump, so goods still in a bag at the end score nothing. A unit carrying goods walks
        to the nearest shed tile once the remaining steps equal its distance, then DROPs (if the shed has room; on step
        718 always). Goods dropped on step 718 are added to the final sell orders."""
        if step < 696 or step > 718:
            return act
        me = int(obs.get("player", 0) or 0)
        f = obs["farms"][me]
        bags = (obs.get("private") or {}).get("inventories") or []
        pos = [f["farmer"]] + list(f["hands"])
        units = [act["farmer"]] + list(act["hands"])
        shed = np.asarray(self._proj_shed, np.int64) if getattr(self, "_proj_shed", None) is not None else None
        room = 100 - int(shed.sum()) if shed is not None else 0
        dropped = {}
        for i, (x, y) in enumerate(pos):
            if i >= len(units) or i >= len(bags):
                continue
            bag = {k: int(v) for k, v in bags[i].items() if int(v) > 0 and k in PRODUCTS}
            if not bag:
                continue
            d = min(abs(x - sx) + abs(y - sy) for sx, sy in SHED_XY)
            if step < 718 - d:
                continue
            u = units[i]
            if d == 0:
                n = sum(bag.values())
                if u and u[0] == "DROP":
                    continue
                if n > room and step < 718:
                    continue
                units[i] = ["DROP"]
                room -= n
                for k, v in bag.items():
                    dropped[k] = dropped.get(k, 0) + v
            else:
                tx, ty = min(SHED_XY, key=lambda s: abs(x - s[0]) + abs(y - s[1]))
                mv = "EAST" if x < tx else "WEST" if x > tx else "SOUTH" if y < ty else "NORTH"
                if not (u and u[0] == mv):
                    units[i] = [mv]
        act["farmer"], act["hands"] = units[0], units[1:]
        if step == 718 and dropped:
            orders = act["market"]
            for k, v in dropped.items():
                hit = [o for o in orders if o[0] == "SELL" and o[1] == k]
                if hit:
                    hit[0][2] = int(hit[0][2]) + v
                elif len(orders) < 10:
                    orders.insert(0, ["SELL", k, v])
        return act
