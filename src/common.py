"""Shared by training and inference: constants, quantity bins, obs dict -> raw arrays, raw -> model features.

Everything here is pure numpy and batched. Raw arrays are seat-normalised (own side first):
  tiles  u8 [B,2,100,10]   upos u8 [B,2,24,2] (255 = absent)   nunit u8 [B,2]
  inv    u8 [B,24,12]      shed u8 [B,12]    seeds i16 [B,5]
  money  f32 [B,2]         hires u8 [B,2]    quads u8 [B,2]
  mkt_inv i16 [B,9] (I0 - inventory)   mkt_price i16 [B,9]   shops u8 [B,8]   step i16 [B]
"""
import datetime

import numpy as np

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
ITEMS = PRODUCTS + ANIMALS
ITEM_ID = {k: i for i, k in enumerate(ITEMS)}
CROPS = PRODUCTS[:5]
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
SHOP_ID = {k: i for i, k in enumerate(SHOPS)}
SHOP_ITEMS = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
              "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
              "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
              "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
# Units each shop consumes per product every 4 steps (single-product shops consume 2) -> [8 shops, 9 products]
SHOP_DEM = np.zeros((8, 9), np.float32)
for _s, _its in SHOP_ITEMS.items():
    for _it in _its:
        SHOP_DEM[SHOP_ID[_s], ITEM_ID[_it]] = 2.0 if len(_its) == 1 else 1.0
BASE = np.array([25, 35, 60, 120, 250, 50, 160, 200, 100], np.float32)
MKT_T = np.array([400, 450, 200, 100, 300, 332, 122, 105, 200], np.float32)
KIND = {"PLANT": 3, "COOP": 4, "PASTURE": 5, "WEED": 2}
UNIT_OPS = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLANT", "WATER", "HARVEST", "FERTILIZE",
            "BUILD_COOP", "BUILD_PASTURE", "DIG", "PLACE", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]
TASK_OPS = ["PASS", "PICKUP", "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "DIG",
            "PLACE", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]
MAXU = 24
NTILE_F = 10

# Date buckets: one bucket per calendar day of the replay, counted from DATE0. The bucket is a conditioning
# input, so at inference time the model can be asked to play like a team's version from a given day.
DATE0 = datetime.date(2026, 8, 17)


def date_bucket(day):
    return max(0, (datetime.date.fromisoformat(day) - DATE0).days)


# Quantity bins: one bin per integer up to 30, coarser above.
QTY_BINS = np.array(list(range(0, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100], np.int32)          # PICKUP / PLACE, SELL
BUY_BINS = np.array(list(range(0, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500, 1000], np.int32)
NHIRE = 16
NANIM = 9

# 2-D axial RoPE. Half of each head's rotation pairs rotate by the tile column x, the other half by the row y,
# so attention scores depend on the relative offset (dx, dy). Rows 0..99 of the table are tiles p
# (x = p % 10, y = p // 10, both farms share coordinates); row 100 is "no rotation" (global, product tokens,
# absent units). Torch training and numpy inference share this table.
ROPE_NOPOS = 100


def rope_table(dh, theta):
    dim = dh // 2
    inv = 1.0 / (float(theta) ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    p = np.arange(100)
    freqs = np.concatenate([(p % 10)[:, None] * inv, (p // 10)[:, None] * inv], 1)
    freqs = np.concatenate([freqs, np.zeros((1, dim))], 0)
    emb = np.concatenate([freqs, freqs], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rope_pos(unit_pos):
    """Position index per token [B, S]: global + 12 products -> no rotation, units -> their tile, tiles -> tile id."""
    B = unit_pos.shape[0]
    return np.concatenate([np.full((B, 13), ROPE_NOPOS, np.int64), np.asarray(unit_pos, np.int64),
                           np.broadcast_to(np.arange(200) % 100, (B, 200))], 1)


def to_bin(v, bins):
    """Nearest bin index."""
    v = np.asarray(v)
    idx = np.searchsorted(bins, v, side="left")
    idx = np.clip(idx, 0, len(bins) - 1)
    lo = np.clip(idx - 1, 0, len(bins) - 1)
    pick_lo = np.abs(bins[lo] - v) < np.abs(bins[idx] - v)
    return np.where(pick_lo, lo, idx)


def enc_tile(tile, day, step, out):
    """Tile dict -> 10 small integers: kind, crop/animal, age, yield, watered/fed, days unwatered/unfed,
    fertilizer days left (plants) or fertilizer available (animals), steps until decay, cared, care bonus."""
    if tile is None:
        return
    if tile == "LOCKED":
        out[0] = 1
        return
    k = tile.get("kind")
    out[0] = KIND.get(k, 0)
    if k == "PLANT":
        out[1] = CROPS.index(tile["crop"]) + 1
        out[2] = max(0, min(255, day - tile["planted_day"]))
        out[3] = tile.get("yield_units", 0)
        out[4] = 1 if tile.get("watered_today") else 0
        out[5] = tile.get("consecutive_unwatered", 0)
        out[6] = max(0, min(3, tile.get("fertilized_until_day", -1) - day + 1))
        mls = tile.get("max_lifespan_step", -1)
        out[7] = 255 if mls < 0 else max(0, min(254, mls - step))
    elif "animal" in tile:
        out[1] = ANIMALS.index(tile["animal"]) + 1
        out[2] = max(0, min(255, day - tile["placed_day"]))
        out[3] = tile.get("yield_units", 0)
        out[4] = 1 if tile.get("fed_today") else 0
        out[5] = tile.get("consecutive_unfed", 0)
        out[6] = 1 if tile.get("fertilizer_available") else 0
        out[8] = 1 if tile.get("cared_today") else 0
        out[9] = min(255, tile.get("pending_care_bonus", 0))


def obs_to_raw(obs):
    """Live observation -> raw arrays (B = 1). Same encoding as extract.py."""
    me = int(obs["player"])
    step = int(obs.get("step", 0) or 0)
    day = step // 24
    farms = obs["farms"]
    r = dict(tiles=np.zeros((1, 2, 100, NTILE_F), np.uint8), upos=np.full((1, 2, MAXU, 2), 255, np.uint8),
             nunit=np.zeros((1, 2), np.uint8), inv=np.zeros((1, MAXU, 12), np.uint8), shed=np.zeros((1, 12), np.uint8),
             seeds=np.zeros((1, 5), np.int16), money=np.zeros((1, 2), np.float32), hires=np.zeros((1, 2), np.uint8),
             quads=np.zeros((1, 2), np.uint8), mkt_inv=np.zeros((1, 9), np.int16), mkt_price=np.zeros((1, 9), np.int16),
             shops=np.zeros((1, 8), np.uint8), step=np.array([step], np.int16))
    for side, f in enumerate((me, 1 - me)):
        farm = farms[f]
        for y in range(10):
            row = farm["tiles"][y]
            for x in range(10):
                enc_tile(row[x], day, step, r["tiles"][0, side, y * 10 + x])
        r["money"][0, side] = farm["money"]
        r["hires"][0, side] = farm.get("hires_today", 0)
        r["quads"][0, side] = len(farm.get("unlocked_quadrants", ["NW"]))
        us = [farm["farmer"]] + list(farm.get("hands", []))
        r["nunit"][0, side] = min(len(us), MAXU)
        for i, p in enumerate(us[:MAXU]):
            r["upos"][0, side, i] = p
    m = obs["market"]
    for i, k in enumerate(PRODUCTS):
        r["mkt_inv"][0, i] = 10000 - m["inventory"][k]
        r["mkt_price"][0, i] = m["prices"][k]
    for s in obs["town"].get("unlocked_shops", []):
        r["shops"][0, SHOP_ID[s]] += 1
    pv = obs["private"]
    for k, v in pv["shed"].items():
        r["shed"][0, ITEM_ID[k]] = min(255, v)
    for i, k in enumerate(CROPS):
        r["seeds"][0, i] = min(32000, pv["seeds"].get(k, 0))
    for i, iv in enumerate(pv["inventories"][:MAXU]):
        for k, v in iv.items():
            if k in ITEM_ID:
                r["inv"][0, i, ITEM_ID[k]] = min(255, v)
    return r


def raw_from_ep(ep, t, seat):
    """Episode arrays (extract.py) + index arrays t, seat of equal length -> batched raw."""
    o = 1 - seat
    return dict(
        tiles=np.stack([ep["tile"][t, seat], ep["tile"][t, o]], 1),
        upos=np.stack([ep["upos"][t, seat], ep["upos"][t, o]], 1),
        nunit=np.stack([ep["nunit"][t, seat], ep["nunit"][t, o]], 1),
        inv=ep["inv"][t, seat], shed=ep["shed"][t, seat], seeds=ep["seeds"][t, seat],
        money=np.stack([ep["money"][t, seat], ep["money"][t, o]], 1),
        hires=np.stack([ep["hires"][t, seat], ep["hires"][t, o]], 1),
        quads=np.stack([ep["quads"][t, seat], ep["quads"][t, o]], 1),
        mkt_inv=ep["mkt_inv"][t], mkt_price=ep["mkt_price"][t], shops=ep["shops"][t], step=t.astype(np.int16))


def life_bucket(v):
    v = v.astype(np.int32)
    b = np.where(v < 48, v, 48 + np.minimum((v - 48) // 24, 14))
    return np.where(v == 255, 63, b)


NITEM_F = 14
NGLOB_F = 24


def featurize(r, team, won, date):
    """Raw batch -> model inputs. team / won / date are int arrays [B]."""
    B = r["tiles"].shape[0]
    tl = r["tiles"].astype(np.int32)
    kindsub = tl[..., 0] * 6 + tl[..., 1]
    tile_ids = np.stack([kindsub, np.minimum(tl[..., 2], 31), np.minimum(tl[..., 3], 7), tl[..., 4] & 1,
                         np.minimum(tl[..., 5], 3), np.minimum(tl[..., 6], 3), life_bucket(tl[..., 7]),
                         tl[..., 8] & 1, np.minimum(tl[..., 9], 15)], -1).reshape(B, 200, 9)
    # units standing on each tile, per side
    ucnt = np.zeros((B, 2, 100), np.int32)
    up = r["upos"].astype(np.int32)
    valid = up[..., 0] < 10
    pid = np.where(valid, up[..., 1] * 10 + up[..., 0], 0)
    bi, si, ui = np.nonzero(valid)
    np.add.at(ucnt, (bi, si, pid[bi, si, ui]), 1)
    tile_ucnt = np.minimum(ucnt, 3).reshape(B, 200)
    unit_pos = np.where(valid[:, 0], pid[:, 0], 100)
    unit_mask = valid[:, 0]
    unit_f = np.log1p(r["inv"].astype(np.float32)) / 3.0
    # product / animal tokens
    own_k, own_s = tl[:, 0, :, 0], tl[:, 0, :, 1]
    opp_k, opp_s = tl[:, 1, :, 0], tl[:, 1, :, 1]
    crop_own = np.stack([((own_k == 3) & (own_s == c + 1)).sum(1) for c in range(5)], 1).astype(np.float32)
    crop_opp = np.stack([((opp_k == 3) & (opp_s == c + 1)).sum(1) for c in range(5)], 1).astype(np.float32)
    ani_own = np.stack([((own_k >= 4) & (own_s == a + 1)).sum(1) for a in range(3)], 1).astype(np.float32)
    ani_opp = np.stack([((opp_k >= 4) & (opp_s == a + 1)).sum(1) for a in range(3)], 1).astype(np.float32)
    dem = r["shops"].astype(np.float32) @ SHOP_DEM
    item_f = np.zeros((B, 12, NITEM_F), np.float32)
    price = r["mkt_price"].astype(np.float32)
    item_f[:, :9, 0] = price / BASE
    item_f[:, :9, 1] = np.clip(r["mkt_inv"].astype(np.float32) / MKT_T, -5, 5)
    item_f[:, :9, 2] = np.log1p(price) / 6
    item_f[:, :9, 3] = dem / 4
    shed = r["shed"].astype(np.float32)
    item_f[:, :, 4] = shed / 20
    item_f[:, :, 5] = np.log1p(shed) / 3
    bag = r["inv"].astype(np.float32).sum(1)
    item_f[:, :, 6] = np.log1p(bag) / 3
    item_f[:, :5, 7] = np.log1p(r["seeds"].astype(np.float32)) / 3
    item_f[:, :5, 8] = crop_own / 20
    item_f[:, :5, 9] = crop_opp / 20
    item_f[:, 9:, 8] = ani_own / 5
    item_f[:, 9:, 9] = ani_opp / 5
    item_f[:, :9, 10] = shed[:, :9] * price / 5000
    item_f[:, :, 11] = shed.sum(1, keepdims=True) / 100
    item_f[:, 9:, 12] = 1.0
    item_f[:, :9, 13] = np.clip(r["mkt_inv"].astype(np.float32) / 100, -10, 10)
    # global token
    step = r["step"].astype(np.int32)
    money = r["money"].astype(np.float32)
    g = np.zeros((B, NGLOB_F), np.float32)
    g[:, 0] = money[:, 0] / 1e4
    g[:, 1] = money[:, 1] / 1e4
    g[:, 2] = np.log1p(np.maximum(money[:, 0], 0)) / 10
    g[:, 3] = np.log1p(np.maximum(money[:, 1], 0)) / 10
    g[:, 4] = r["hires"][:, 0] / 10
    g[:, 5] = r["hires"][:, 1] / 10
    g[:, 6] = r["quads"][:, 0] / 4
    g[:, 7] = r["quads"][:, 1] / 4
    g[:, 8] = r["nunit"][:, 0] / 10
    g[:, 9] = r["nunit"][:, 1] / 10
    g[:, 10:18] = r["shops"] / 2
    g[:, 18] = step / 720
    g[:, 19] = r["shops"].sum(1) / 8
    g[:, 20] = shed.sum(1) / 100
    g[:, 21] = (money[:, 0] - money[:, 1]) / 1e4
    g[:, 22] = np.minimum(money[:, 0], 3000) / 3000
    g[:, 23] = ((step % 24) == 0)
    glob_ids = np.stack([step % 24, np.minimum(step // 24, 29), np.asarray(team, np.int32),
                         np.asarray(won, np.int32), np.asarray(date, np.int32)], -1)
    return dict(tile_ids=tile_ids.astype(np.int16), tile_ucnt=tile_ucnt.astype(np.int8),
                unit_pos=unit_pos.astype(np.int16), unit_mask=unit_mask, unit_f=unit_f,
                item_f=item_f, glob_f=g, glob_ids=glob_ids.astype(np.int16))
