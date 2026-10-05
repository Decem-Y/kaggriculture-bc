"""BC data: load episode / label npz files and draw (step, seat) samples from a rotating pool of episodes.

Each DataLoader worker featurises its own batches; the main process only moves tensors.
"""
import json, os

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from common import raw_from_ep, featurize, to_bin, date_bucket, QTY_BINS, BUY_BINS, NHIRE, NANIM

EP_KEYS = ["tile", "upos", "nunit", "inv", "shed", "seeds", "money", "hires", "quads", "mkt_inv", "mkt_price", "shops", "uact"]
LAB_KEYS = ["u_op", "u_tile", "u_item", "u_qty", "m_sell", "m_seed", "m_buyp", "m_anim", "m_hire", "m_land"]


def build_index(census_path, ep_dir, lab_dir, vocab, since="2000-01-01", val_mod=50, ndate=None):
    """Census rows -> episode records. A seat is trained only if its census mask is True; won is 1 / 0 / 2 (tie).
    Episodes with episode_id % val_mod == 0 form the validation split."""
    tid = {t: i for i, t in enumerate(vocab)}
    eps = []
    for line in open(census_path):
        r = json.loads(line)
        if r["day"] < since:
            continue
        ep = os.path.join(ep_dir, f"{r['id']}.npz")
        lab = os.path.join(lab_dir, f"{r['id']}.npz")
        if not (os.path.exists(ep) and os.path.exists(lab)):
            continue
        mask = r.get("mask", [True, True])
        if not any(mask):
            continue
        rw = r["rewards"]
        won = [2 if rw[0] == rw[1] else int(rw[s] > rw[1 - s]) for s in range(2)]
        seat_date = [date_bucket(d) for d in r.get("seat_days", [r["day"], r["day"]])]
        if ndate:
            seat_date = [min(d, ndate - 1) for d in seat_date]
        eps.append(dict(id=r["id"], day=r["day"], ep=ep, lab=lab, team=[tid.get(t, 0) for t in r["teams"]],
                        won=won, seat_date=seat_date, mask=list(mask), val=(r["id"] % val_mod == 0)))
    return eps


def load_episode(e):
    a = np.load(e["ep"])
    b = np.load(e["lab"])
    d = {k: a[k] for k in EP_KEYS}
    d.update({k: b[k] for k in LAB_KEYS})
    return d


def make_batch(d, t, seat, team, won, date):
    r = raw_from_ep(d, t, seat)
    x = featurize(r, team, won, date)
    y = dict(
        u_op=d["u_op"][t, seat].astype(np.int64),
        u_tile=d["u_tile"][t, seat].astype(np.int64),
        u_item=d["u_item"][t, seat].astype(np.int64),
        u_qty=to_bin(np.minimum(d["u_qty"][t, seat], 100), QTY_BINS).astype(np.int64),
        m_sell=to_bin(np.minimum(d["m_sell"][t, seat], 100), QTY_BINS).astype(np.int64),
        m_seed=to_bin(np.minimum(d["m_seed"][t, seat], 1000), BUY_BINS).astype(np.int64),
        m_buyp=to_bin(np.minimum(d["m_buyp"][t, seat], 1000), BUY_BINS).astype(np.int64),
        m_anim=np.minimum(d["m_anim"][t, seat], NANIM - 1).astype(np.int64),
        m_hire=np.minimum(d["m_hire"][t, seat], NHIRE - 1).astype(np.int64),
        m_land=d["m_land"][t, seat].astype(np.int64),
    )
    # Raw unit actions: op 0..17; item only for PICKUP / PLANT / PLACE; quantity bin only for PICKUP / PLACE.
    # A unit is valid when it has a task label (present at step t).
    ua = d["uact"][t, seat].astype(np.int64)
    valid = y["u_op"] >= 0
    rop = np.where(valid, ua[..., 0], -1)
    has_it = (rop == 5) | (rop == 6) | (rop == 13)
    y["r_op"] = rop
    y["r_item"] = np.where(has_it, np.clip(ua[..., 1], 0, 12), 0)
    y["r_qty"] = to_bin(np.clip(ua[..., 2], 0, 100), QTY_BINS).astype(np.int64)
    return x, y


def concat(parts):
    xs, ys = zip(*parts)
    x = {k: np.concatenate([p[k] for p in xs]) for k in xs[0]}
    y = {k: np.concatenate([p[k] for p in ys]) for k in ys[0]}
    return x, y


class EpisodePool(IterableDataset):
    """Keeps `pool` episodes in memory, each with a shuffled list of its (step, seat) samples. A batch draws episodes
    uniformly from the pool; an exhausted episode is replaced by a new draw (weighted by `weights` if given)."""

    def __init__(self, eps, batch_size, pool=48, seed=0, seat_filter=None, weights=None):
        self.eps, self.bs, self.pool, self.seed = eps, batch_size, pool, seed
        self.seat_filter = seat_filter      # fn(e, seat) -> bool
        self.weights = weights

    def _samples(self, e):
        ts = np.arange(719)
        out = []
        for s in range(2):
            if e["mask"][s] and (self.seat_filter is None or self.seat_filter(e, s)):
                out.append(np.stack([ts, np.full_like(ts, s)], 1))
        if not out:
            return None
        return np.concatenate(out)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng(self.seed * 1000 + wid)
        mine = [e for i, e in enumerate(self.eps) if i % nw == wid]
        p = None
        if self.weights is not None:
            w = np.array([self.weights[i] for i, e in enumerate(self.eps) if i % nw == wid], np.float64)
            p = w / w.sum()

        def draw():
            while True:
                e = mine[rng.choice(len(mine), p=p)]
                sm = self._samples(e)
                if sm is None:
                    continue
                try:
                    d = load_episode(e)
                except Exception:
                    continue
                return [e, d, sm[rng.permutation(len(sm))], 0]

        pool = [draw() for _ in range(min(self.pool, len(mine)))]
        while True:
            which = rng.integers(0, len(pool), self.bs)
            parts = []
            for k in np.unique(which):
                n = int((which == k).sum())
                e, d, sm, ptr = pool[k]
                take = sm[ptr:ptr + n]
                pool[k][3] = ptr + n
                t, s = take[:, 0], take[:, 1]
                team = np.array(e["team"])[s]
                won = np.array(e["won"])[s]
                parts.append(make_batch(d, t, s, team, won, np.asarray(e["seat_date"])[s]))
                if pool[k][3] >= len(sm):
                    pool[k] = draw()
            x, y = concat(parts)
            yield ({k: torch.from_numpy(v) for k, v in x.items()}, {k: torch.from_numpy(v) for k, v in y.items()})


def loader(eps, batch_size, workers=12, pool=48, seed=0, seat_filter=None, weights=None):
    ds = EpisodePool(eps, batch_size, pool=pool, seed=seed, seat_filter=seat_filter, weights=weights)
    return DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      prefetch_factor=4 if workers > 0 else None)


def fixed_val(eps, n_eps=64, per_ep=256, seed=1, seat_filter=None):
    """Fixed validation set: the first n_eps episodes, per_ep uniformly drawn (step, seat) samples each."""
    rng = np.random.default_rng(seed)
    parts = []
    for e in eps[:n_eps]:
        d = load_episode(e)
        ts = rng.integers(0, 719, per_ep)
        ss = rng.integers(0, 2, per_ep)
        if seat_filter is not None or not all(e["mask"]):
            keep = np.array([e["mask"][int(s)] and (seat_filter is None or seat_filter(e, int(s))) for s in ss])
            ts, ss = ts[keep], ss[keep]
            if len(ts) == 0:
                continue
        team = np.array(e["team"])[ss]
        won = np.array(e["won"])[ss]
        parts.append(make_batch(d, ts, ss, team, won, np.asarray(e["seat_date"])[ss]))
    return concat(parts)
