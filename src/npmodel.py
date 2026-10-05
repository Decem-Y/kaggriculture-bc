"""Pure-numpy forward pass of BCNet, op-for-op equal to model.py, so the submission depends on numpy only.
Weights come from export.py."""
import math
import numpy as np

from common import MAXU, rope_table, rope_pos

I_ITEM, I_UNIT, I_TILE = 1, 13, 13 + MAXU


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


class NPNet:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.W = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("_")}
        self.layers = int(z["_layers"])
        self.heads = int(z["_heads"])
        self.d = self.W["e_type.weight"].shape[1]
        self.capv = float(z["_cap"]) if "_cap" in z.files else 0.0
        self.has_date = "e_date.weight" in self.W
        self.rope = float(z["_rope"]) if "_rope" in z.files else 0.0
        if self.rope:
            self.rcos, self.rsin = rope_table(self.d // self.heads, self.rope)

    def _cap(self, x):
        return self.capv * np.tanh(x / self.capv) if self.capv > 0 else x

    def lin(self, x, name):
        return x @ self.W[name + ".weight"].T + self.W[name + ".bias"]

    def emb(self, name, idx):
        return self.W[name + ".weight"][idx]

    def encode(self, x):
        W = self.W
        ti = x["tile_ids"].astype(np.int64)
        B = ti.shape[0]
        side = np.arange(200) // 100
        tpos = np.arange(200) % 100
        tile = (self.emb("e_kindsub", ti[..., 0]) + self.emb("e_age", ti[..., 1]) + self.emb("e_yield", ti[..., 2])
                + self.emb("e_wet", ti[..., 3]) + self.emb("e_cons", ti[..., 4]) + self.emb("e_fert", ti[..., 5])
                + self.emb("e_life", ti[..., 6]) + self.emb("e_cared", ti[..., 7]) + self.emb("e_cb", ti[..., 8])
                + self.emb("e_tpos", tpos) + self.emb("e_side", side) + self.emb("e_ucnt", x["tile_ucnt"].astype(np.int64))
                + W["e_type.weight"][3])
        unit = (self.emb("e_upos", x["unit_pos"].astype(np.int64)) + W["e_uidx.weight"][None]
                + self.lin(x["unit_f"], "unit_lin") + W["e_type.weight"][2])
        item = W["e_item.weight"][None] + self.lin(x["item_f"], "item_lin") + W["e_type.weight"][1]
        gi = x["glob_ids"].astype(np.int64)
        glob = (self.lin(x["glob_f"], "glob_lin") + self.emb("e_hour", gi[:, 0]) + self.emb("e_day", gi[:, 1])
                + self.emb("e_team", gi[:, 2]) + self.emb("e_won", gi[:, 3]) + W["e_type.weight"][0])
        if self.has_date:
            glob = glob + self.emb("e_date", gi[:, 4])
        h = np.concatenate([glob[:, None], np.broadcast_to(item, (B,) + item.shape[1:]), unit, tile], 1).astype(np.float32)
        S = h.shape[1]
        H, D = self.heads, self.d
        dh = D // H
        if self.rope:
            pos = rope_pos(x["unit_pos"])
            cos, sin = self.rcos[pos][:, None], self.rsin[pos][:, None]
            rot = lambda t: np.concatenate([-t[..., dh // 2:], t[..., :dh // 2]], -1)
        for l in range(self.layers):
            p = f"blocks.{l}."
            qkv = self.lin(_ln(h, W[p + "ln1.weight"], W[p + "ln1.bias"]), p + "qkv").reshape(B, S, 3, H, dh)
            q, k, v = qkv[:, :, 0].transpose(0, 2, 1, 3), qkv[:, :, 1].transpose(0, 2, 1, 3), qkv[:, :, 2].transpose(0, 2, 1, 3)
            if self.rope:
                q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
            att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(dh))
            att = att - att.max(-1, keepdims=True)
            att = np.exp(att)
            att /= att.sum(-1, keepdims=True)
            y = (att @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
            h = h + self.lin(y, p + "proj")
            h = h + self.lin(_gelu(self.lin(_ln(h, W[p + "ln2.weight"], W[p + "ln2.bias"]), p + "fc1")), p + "fc2")
        return _ln(h, W["ln_f.weight"], W["ln_f.bias"])

    # ---- unit decoder, one unit at a time (B = 1) ----
    def unit_ctx(self, h):
        """Per-step decoding context."""
        W = self.W
        hu = h[0, I_UNIT:I_TILE]
        ht = h[0, I_TILE:I_TILE + 100]
        return dict(hu=hu, ht=ht, k=self.lin(ht, "ptr_k"), mem_u=self.lin(hu, "mem_u"), mem_t=self.lin(ht, "mem_t"),
                    kv=[], mem=[W["m_null"][None].astype(np.float32)])

    def _dec_layer(self, l, z, m):
        W = self.W
        p = f"dec.{l}."
        H, D = self.heads, self.d
        dh = D // H
        q = self.lin(_ln(z, W[p + "ln_q.weight"], W[p + "ln_q.bias"]), p + "q").reshape(H, dh)
        kv = self.lin(_ln(m, W[p + "ln_m.weight"], W[p + "ln_m.bias"]), p + "kv").reshape(len(m), 2, H, dh)
        k, v = kv[:, 0].transpose(1, 0, 2), kv[:, 1].transpose(1, 0, 2)
        att = (k @ q[:, :, None])[..., 0] * (1.0 / math.sqrt(dh))
        att = att - att.max(-1, keepdims=True)
        att = np.exp(att)
        att /= att.sum(-1, keepdims=True)
        y = (att[:, None] @ v)[:, 0].reshape(D)
        z = z + self.lin(y, p + "proj")
        return z + self.lin(_gelu(self.lin(_ln(z, W[p + "ln2.weight"], W[p + "ln2.bias"]), p + "fc1")), p + "fc2")

    def unit_query(self, ctx, i, pos):
        """Decoder output zf [d] for unit i standing on tile pos; the memory holds null + earlier units only."""
        W = self.W
        z = ctx["hu"][i] + self.lin(ctx["ht"][min(pos, 99)], "cur_tile")
        m = np.concatenate(ctx["mem"], 0)
        l = 0
        while f"dec.{l}.q.weight" in W:
            z = self._dec_layer(l, z, m)
            l += 1
        return _ln(z, W["ln_d.weight"], W["ln_d.bias"])

    def unit_scores(self, ctx, zf):
        """Task log-probs [14] and per-task tile log-probs [14, 100]."""
        W = self.W
        lop = self._cap(self.lin(zf, "op_head"))
        qa = self.lin(zf, "ptr_q")[None] + W["e_opq.weight"]
        lt = self._cap((qa @ ctx["k"].T) / math.sqrt(self.d))
        lsm = lambda x: x - x.max(-1, keepdims=True) - np.log(np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True))
        return lsm(lop), lsm(lt)

    def raw_logits(self, ctx, zf, op, tile, pos):
        W = self.W
        tc, up = int(tile), min(int(pos), 99)
        rdx = max(-9, min(9, tc % 10 - up % 10)) + 9
        rdy = max(-9, min(9, tc // 10 - up // 10)) + 9
        a = _gelu(self.lin(zf, "raw_fc") + self.lin(ctx["ht"][tc], "raw_tile") + W["e_rop.weight"][op]
                  + W["e_rdx.weight"][rdx] + W["e_rdy.weight"][rdy])
        return self._cap(self.lin(a, "raw_out"))

    def arg_logits(self, ctx, zf, raw, tile):
        W = self.W
        b = _gelu(self.lin(zf + W["e_rawa.weight"][raw], "arg_fc") + self.lin(ctx["ht"][int(tile)], "arg_tile"))
        return self._cap(self.lin(b, "item_head")), self._cap(self.lin(b, "qty_head"))

    def push_memory(self, ctx, i, op, tile, raw, item):
        W = self.W
        m = (ctx["mem_u"][i] + W["e_mop.weight"][op] + ctx["mem_t"][int(tile)] + W["e_mraw.weight"][raw]
             + W["e_mitem.weight"][max(0, min(12, item))])
        ctx["mem"].append(m[None].astype(np.float32))

    def market(self, h):
        hi = h[:, I_ITEM:I_UNIT]
        g = h[:, 0]
        L = lambda x, n: self._cap(self.lin(x, n))
        return dict(sell=L(hi[:, :9], "sell_head"), seed=L(hi[:, :5], "seed_head"),
                    buyp=L(hi[:, [0, 8]], "buyp_head"), anim=L(hi[:, 9:12], "anim_head"),
                    hire=L(g, "hire_head"), land=L(g, "land_head"))
