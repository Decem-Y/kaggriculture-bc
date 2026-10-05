"""Replay JSON -> one compact npz per episode. Lossless encoding only; labels are derived in labels.py.

Index t = 0..719 is the observation of steps[t]; the action taken from it is steps[t+1][seat].action, so t = 719
has no action.
"""
import json, os

import numpy as np

from common import PRODUCTS, ITEM_ID, CROPS, SHOP_ID, UNIT_OPS, MAXU, NTILE_F, enc_tile

UOP = {k: i for i, k in enumerate(UNIT_OPS)}
MKT_OPS = ["NONE", "SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND"]
MOP = {k: i for i, k in enumerate(MKT_OPS)}
MAXO = 10      # the engine executes only the first 10 market orders
T = 720


def enc_unit_action(a):
    if not isinstance(a, list) or not a:
        return (0, 0, 0)
    op = UOP.get(a[0], 0)
    item, n = 0, 0
    if len(a) >= 2 and isinstance(a[1], str):
        item = ITEM_ID.get(a[1], -1) + 1
    if len(a) >= 3:
        try:
            n = int(a[2])
        except Exception:
            n = 0
    elif op in (UOP["PICKUP"], UOP["PLACE"]):
        n = 1
    return (op, item, max(-1, min(n, 30000)))


def enc_order(o):
    if not isinstance(o, list) or not o:
        return (0, 0, 0)
    op = MOP.get(o[0], 0)
    item, n = 0, 0
    if len(o) >= 2 and isinstance(o[1], str):
        item = ITEM_ID.get(o[1], -1) + 1
    if len(o) >= 3:
        try:
            n = int(o[2])
        except Exception:
            n = 0
    return (op, item, max(-1, min(n, 2_000_000_000)))


def extract(path):
    with open(path, "rb") as f:
        d = json.load(f)
    steps = d["steps"]
    assert len(steps) == T, len(steps)
    tile = np.zeros((T, 2, 100, NTILE_F), np.uint8)
    money = np.zeros((T, 2), np.float32)
    hires = np.zeros((T, 2), np.uint8)
    quads = np.zeros((T, 2), np.uint8)
    upos = np.full((T, 2, MAXU, 2), 255, np.uint8)
    nunit = np.zeros((T, 2), np.uint8)
    mkt_inv = np.zeros((T, 9), np.int16)
    mkt_price = np.zeros((T, 9), np.int16)
    shops = np.zeros((T, 8), np.uint8)
    shed = np.zeros((T, 2, 12), np.uint8)
    seeds = np.zeros((T, 2, 5), np.int16)
    inv = np.zeros((T, 2, MAXU, 12), np.uint8)
    uact = np.zeros((T, 2, MAXU, 3), np.int16)
    mact = np.zeros((T, 2, MAXO, 3), np.int32)
    nord = np.zeros((T, 2), np.uint8)
    for t in range(T):
        o0 = steps[t][0]["observation"]
        day, step = o0["day"], t
        for f, farm in enumerate(o0["farms"]):
            tl = farm["tiles"]
            for y in range(10):
                row = tl[y]
                for x in range(10):
                    enc_tile(row[x], day, step, tile[t, f, y * 10 + x])
            money[t, f] = farm["money"]
            hires[t, f] = farm.get("hires_today", 0)
            quads[t, f] = len(farm.get("unlocked_quadrants", ["NW"]))
            us = [farm["farmer"]] + list(farm.get("hands", []))
            nunit[t, f] = min(len(us), MAXU)
            for i, p in enumerate(us[:MAXU]):
                upos[t, f, i] = p
        m = o0["market"]
        for i, k in enumerate(PRODUCTS):
            mkt_inv[t, i] = 10000 - m["inventory"][k]
            mkt_price[t, i] = m["prices"][k]
        for s in o0["town"].get("unlocked_shops", []):
            shops[t, SHOP_ID[s]] += 1
        for seat in range(2):
            pv = steps[t][seat]["observation"]["private"]
            for k, v in pv["shed"].items():
                shed[t, seat, ITEM_ID[k]] = min(255, v)
            for i, k in enumerate(CROPS):
                seeds[t, seat, i] = min(32000, pv["seeds"].get(k, 0))
            for i, iv in enumerate(pv["inventories"][:MAXU]):
                for k, v in iv.items():
                    if k in ITEM_ID:
                        inv[t, seat, i, ITEM_ID[k]] = min(255, v)
            if t + 1 < T:
                a = steps[t + 1][seat]["action"] or {}
                if not isinstance(a, dict):
                    a = {}
                uact[t, seat, 0] = enc_unit_action(a.get("farmer", ["PASS"]))
                hs = a.get("hands", []) or []
                if isinstance(hs, list):
                    for i, h in enumerate(hs[:MAXU - 1]):
                        uact[t, seat, i + 1] = enc_unit_action(h)
                ms = a.get("market", []) or []
                if isinstance(ms, list):
                    nord[t, seat] = min(len(ms), 255)
                    for i, od in enumerate(ms[:MAXO]):
                        mact[t, seat, i] = enc_order(od)
    info = d.get("info", {})
    meta = dict(id=info.get("EpisodeId"), seed=info.get("seed"), teams=info.get("TeamNames"),
                rewards=d.get("rewards"), statuses=d.get("statuses"))
    return dict(tile=tile, money=money, hires=hires, quads=quads, upos=upos, nunit=nunit, mkt_inv=mkt_inv,
                mkt_price=mkt_price, shops=shops, shed=shed, seeds=seeds, inv=inv, uact=uact, mact=mact,
                nord=nord, meta=np.array(json.dumps(meta)))


def extract_to(args):
    """(replay path, output npz) -> status string. Skips existing outputs; writes atomically."""
    path, out = args
    if os.path.exists(out):
        return "skip"
    try:
        arr = extract(path)
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, out)
        return "ok"
    except Exception as e:
        return f"err {out} {e!r:.200}"
