"""Entity Transformer BC policy.

Encoder sequence: [global, 12 product/animal tokens, 24 unit tokens, 100 own tiles, 100 opponent tiles], with 2-D
axial RoPE on tile coordinates.

Unit decoder, one unit at a time (farmer first). The query of unit i is its encoder token plus the token of the tile
it stands on; it cross-attends to [null] + one memory slot per earlier unit j < i (that unit's token plus its chosen
task, target tile, raw action and item). Per unit it outputs, each conditioned on the previous choices:
task (14) -> target tile pointer (100) -> raw action (18, incl. moves) -> item / quantity.
Only the raw action is sent to the engine; task + target tile tell later units which jobs are already claimed.

Market heads read the encoder output directly. Only ops with an exact numpy counterpart are used (explicit
attention, tanh-GELU, LayerNorm) so the forward pass can be re-implemented in npmodel.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common import QTY_BINS, BUY_BINS, NHIRE, NANIM, NITEM_F, NGLOB_F, TASK_OPS, UNIT_OPS, MAXU, rope_table, ROPE_NOPOS

NOP = len(TASK_OPS)
NRAW = len(UNIT_OPS)
NQ, NB = len(QTY_BINS), len(BUY_BINS)
SEQ = 1 + 12 + MAXU + 200
I_ITEM, I_UNIT, I_TILE = 1, 13, 13 + MAXU


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    dt = q.dtype
    q, k = q.float(), k.float()
    return ((q * cos) + (rotate_half(q) * sin)).to(dt), ((k * cos) + (rotate_half(k) * sin)).to(dt)


class Block(nn.Module):
    def __init__(self, d, h, ff, drop, flash=False):
        super().__init__()
        self.h = h
        self.flash = flash
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, cos=None, sin=None):
        B, S, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cos is not None:
            q, k = apply_rope(q, k, cos, sin)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            att = ((q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(D // self.h))).float().softmax(-1).to(v.dtype)
            y = att @ v
        y = y.transpose(1, 2).reshape(B, S, D)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh")))
        return x


class DecLayer(nn.Module):
    """Unit queries [B,U,d] cross-attend to unit memory [B,U+1,d] (slot 0 = null) under mask [U,U+1] (True = visible)."""

    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.ln_q = nn.LayerNorm(d)
        self.ln_m = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)

    def forward(self, z, m, mask):
        B, U, D = z.shape
        S = m.shape[1]
        dh = D // self.h
        q = self.q(self.ln_q(z)).view(B, U, self.h, dh).transpose(1, 2)
        k, v = self.kv(self.ln_m(m)).view(B, S, 2, self.h, dh).permute(2, 0, 3, 1, 4)
        att = ((q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(dh))).float()
        att = att.masked_fill(~mask, -1e9).softmax(-1).to(v.dtype)
        y = (att @ v).transpose(1, 2).reshape(B, U, D)
        z = z + self.proj(y)
        z = z + self.fc2(F.gelu(self.fc1(self.ln2(z)), approximate="tanh"))
        return z


def causal_mask(U, dev):
    """Unit i sees slot 0 (null) and slots j + 1 for j < i."""
    return torch.arange(U + 1, device=dev)[None, :] <= torch.arange(U, device=dev)[:, None]


class BCNet(nn.Module):
    def __init__(self, n_team, d=256, layers=8, heads=8, ff=1024, drop=0.0, ckpt=False, cap=0.0, ndate=0,
                 dec_layers=2, dec_ff=512, rope_theta=0.0, attn="explicit"):
        super().__init__()
        self.rope = float(rope_theta)
        if self.rope:
            c, s_ = rope_table(d // heads, self.rope)
            self.register_buffer("rope_cos", torch.from_numpy(c), persistent=False)
            self.register_buffer("rope_sin", torch.from_numpy(s_), persistent=False)
        self.capv = float(cap)      # > 0: logit soft cap cap * tanh(x / cap)
        self.ndate = int(ndate)     # > 0: date-bucket embedding
        self.d = d
        self.ckpt = ckpt
        E = lambda n: nn.Embedding(n, d)
        self.e_kindsub, self.e_age, self.e_yield, self.e_wet = E(36), E(32), E(8), E(2)
        self.e_cons, self.e_fert, self.e_life, self.e_cared, self.e_cb = E(4), E(4), E(64), E(2), E(16)
        self.e_tpos, self.e_side, self.e_ucnt = E(100), E(2), E(4)
        self.e_upos, self.e_uidx = E(101), E(MAXU)
        self.unit_lin = nn.Linear(12, d)
        self.e_item = E(12)
        self.item_lin = nn.Linear(NITEM_F, d)
        self.glob_lin = nn.Linear(NGLOB_F, d)
        self.e_hour, self.e_day, self.e_team, self.e_won = E(24), E(30), E(n_team), E(3)
        self.e_type = E(4)
        if self.ndate:
            self.e_date = E(self.ndate)
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop, flash=attn == "flash") for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        # unit decoder
        self.cur_tile = nn.Linear(d, d)
        self.mem_u = nn.Linear(d, d)
        self.mem_t = nn.Linear(d, d)
        self.e_mop, self.e_mraw, self.e_mitem = E(NOP), E(NRAW), E(13)
        self.m_null = nn.Parameter(torch.zeros(d))
        self.dec = nn.ModuleList([DecLayer(d, heads, dec_ff) for _ in range(dec_layers)])
        self.ln_d = nn.LayerNorm(d)
        # task / target tile
        self.op_head = nn.Linear(d, NOP)
        self.ptr_q = nn.Linear(d, d)
        self.ptr_k = nn.Linear(d, d)
        self.e_opq = E(NOP)
        # raw action, conditioned on task, target tile and offset to it
        self.raw_fc = nn.Linear(d, d)
        self.raw_tile = nn.Linear(d, d)
        self.e_rop = E(NOP)
        self.e_rdx, self.e_rdy = E(19), E(19)
        self.raw_out = nn.Linear(d, NRAW)
        # item / quantity, conditioned on raw action and target tile
        self.e_rawa = E(NRAW)
        self.arg_fc = nn.Linear(d, d)
        self.arg_tile = nn.Linear(d, d)
        self.item_head = nn.Linear(d, 13)
        self.qty_head = nn.Linear(d, NQ)
        # market
        self.sell_head = nn.Linear(d, NQ)
        self.seed_head = nn.Linear(d, NB)
        self.buyp_head = nn.Linear(d, NB)
        self.anim_head = nn.Linear(d, NANIM)
        self.hire_head = nn.Linear(d, NHIRE)
        self.land_head = nn.Linear(d, 2)
        self.apply(self._init)

    def cap(self, x):
        return self.capv * torch.tanh(x / self.capv) if self.capv > 0 else x

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, x):
        ti = x["tile_ids"].long()
        B = ti.shape[0]
        dev = ti.device
        side = torch.arange(200, device=dev) // 100
        tpos = torch.arange(200, device=dev) % 100
        tile = (self.e_kindsub(ti[..., 0]) + self.e_age(ti[..., 1]) + self.e_yield(ti[..., 2]) + self.e_wet(ti[..., 3])
                + self.e_cons(ti[..., 4]) + self.e_fert(ti[..., 5]) + self.e_life(ti[..., 6]) + self.e_cared(ti[..., 7])
                + self.e_cb(ti[..., 8]) + self.e_tpos(tpos) + self.e_side(side) + self.e_ucnt(x["tile_ucnt"].long())
                + self.e_type.weight[3])
        unit = (self.e_upos(x["unit_pos"].long()) + self.e_uidx.weight[None] + self.unit_lin(x["unit_f"])
                + self.e_type.weight[2])
        item = self.e_item.weight[None] + self.item_lin(x["item_f"]) + self.e_type.weight[1]
        gi = x["glob_ids"].long()
        glob = (self.glob_lin(x["glob_f"]) + self.e_hour(gi[:, 0]) + self.e_day(gi[:, 1]) + self.e_team(gi[:, 2])
                + self.e_won(gi[:, 3]) + self.e_type.weight[0])
        if self.ndate:
            glob = glob + self.e_date(gi[:, 4])
        # No padding mask: absent units sit on position 100 ("no position") and the model learns to ignore them.
        h = torch.cat([glob[:, None], item.expand(B, -1, -1), unit, tile], 1)
        cos = sin = None
        if self.rope:
            pos = torch.cat([torch.full((B, 13), ROPE_NOPOS, dtype=torch.long, device=dev), x["unit_pos"].long(),
                             tpos.expand(B, -1)], 1)
            cos, sin = self.rope_cos[pos][:, None], self.rope_sin[pos][:, None]
        for blk in self.blocks:
            h = checkpoint(blk, h, cos, sin, use_reentrant=False) if (self.ckpt and self.training) else blk(h, cos, sin)
        return self.ln_f(h)

    @staticmethod
    def gather_tiles(ht, idx):
        return torch.gather(ht, 1, idx.long().clamp(0, 99)[..., None].expand(-1, -1, ht.shape[-1]))

    def memory(self, hu, ht, op, tile, raw, item):
        """Per-unit memory [B,U,d]; run_dec prepends the null slot."""
        return (self.mem_u(hu) + self.e_mop(op.clamp(min=0)) + self.mem_t(self.gather_tiles(ht, tile))
                + self.e_mraw(raw.clamp(min=0)) + self.e_mitem(item.clamp(0, 12)))

    def dec_query(self, hu, ht, unit_pos):
        return hu + self.cur_tile(self.gather_tiles(ht, unit_pos))

    def run_dec(self, z, mem):
        B, U, _ = z.shape
        m = torch.cat([self.m_null.to(mem.dtype)[None, None].expand(B, 1, -1), mem], 1)
        mask = causal_mask(U, z.device)
        for lay in self.dec:
            z = lay(z, m, mask)
        return self.ln_d(z)

    def raw_logits(self, zf, ht, op, tile, unit_pos):
        tc = tile.long().clamp(0, 99)
        up = unit_pos.long().clamp(0, 99)
        rdx = (tc % 10 - up % 10).clamp(-9, 9) + 9
        rdy = (tc // 10 - up // 10).clamp(-9, 9) + 9
        a = F.gelu(self.raw_fc(zf) + self.raw_tile(self.gather_tiles(ht, tile)) + self.e_rop(op.clamp(min=0))
                   + self.e_rdx(rdx) + self.e_rdy(rdy), approximate="tanh")
        return self.cap(self.raw_out(a))

    def arg_logits(self, zf, ht, raw, tile):
        b = F.gelu(self.arg_fc(zf + self.e_rawa(raw.clamp(min=0))) + self.arg_tile(self.gather_tiles(ht, tile)), approximate="tanh")
        return self.cap(self.item_head(b)), self.cap(self.qty_head(b))

    def market_heads(self, h):
        hi = h[:, I_ITEM:I_UNIT]
        g = h[:, 0]
        return dict(sell=self.cap(self.sell_head(hi[:, :9])), seed=self.cap(self.seed_head(hi[:, :5])),
                    buyp=self.cap(self.buyp_head(hi[:, [0, 8]])), anim=self.cap(self.anim_head(hi[:, 9:12])),
                    hire=self.cap(self.hire_head(g)), land=self.cap(self.land_head(g)))

    def forward(self, x, op, tile, raw, item):
        """Training forward with teacher forcing: the memory holds the teacher's choices for earlier units."""
        h = self.encode(x)
        hu = h[:, I_UNIT:I_TILE]
        ht = h[:, I_TILE:I_TILE + 100]
        up = x["unit_pos"]
        zf = self.run_dec(self.dec_query(hu, ht, up), self.memory(hu, ht, op, tile, raw, item))
        op_l = self.cap(self.op_head(zf))
        q = self.ptr_q(zf) + self.e_opq(op.clamp(min=0))
        tile_l = self.cap((q @ self.ptr_k(ht).transpose(1, 2)) / math.sqrt(self.d))
        raw_l = self.raw_logits(zf, ht, op, tile, up)
        item_l, qty_l = self.arg_logits(zf, ht, raw, tile)
        return op_l, tile_l, raw_l, item_l, qty_l, self.market_heads(h)

    @torch.no_grad()
    def decode(self, x, raw_mask=None, topk=0):
        """Autoregressive decoding: per unit, joint argmax over (task, tile), then the raw action and its arguments,
        written back to the memory for later units.
        topk > 0: marginalise the raw action over the top-k (task, tile) plans,
        argmax_raw sum_k p(task, tile) * p(raw | task, tile); the plan written to memory is the one with the highest
        p(task, tile) * p(raw* | task, tile).
        raw_mask: optional fn(i, op, tile, raw_logits) -> raw_logits."""
        h = self.encode(x)
        B = h.shape[0]
        hu = h[:, I_UNIT:I_TILE]
        ht = h[:, I_TILE:I_TILE + 100]
        up = x["unit_pos"]
        z0 = self.dec_query(hu, ht, up)
        k = self.ptr_k(ht)
        dev = h.device
        OP = torch.zeros(B, MAXU, dtype=torch.long, device=dev)
        TL = torch.zeros_like(OP)
        RW = torch.zeros_like(OP)
        IT = torch.zeros_like(OP)
        QT = torch.zeros_like(OP)
        for i in range(MAXU):
            zf = self.run_dec(z0, self.memory(hu, ht, OP, TL, RW, IT))[:, i]
            lop = F.log_softmax(self.cap(self.op_head(zf)).float(), -1)
            qa = self.ptr_q(zf)[:, None] + self.e_opq.weight[None]
            lt = F.log_softmax(self.cap((qa @ k.transpose(1, 2)) / math.sqrt(self.d)).float(), -1)
            sc = (lop[..., None] + lt).reshape(B, -1)
            if topk:
                lp, j = sc.topk(topk, -1)
                ops, tls = j // 100, j % 100
                rl = self.raw_logits(zf[:, None].expand(-1, topk, -1), ht, ops, tls, up[:, i:i + 1].expand(-1, topk))
                lr = F.log_softmax(rl.float(), -1)
                marg = torch.logsumexp(lp[..., None] + lr, 1)
                if raw_mask is not None:
                    marg = raw_mask(i, None, None, marg)
                rw_i = marg.argmax(-1)
                post = lp + lr.gather(-1, rw_i[:, None, None].expand(-1, topk, 1))[..., 0]
                kk = post.argmax(-1)
                op_i, tl_i = ops.gather(1, kk[:, None])[:, 0], tls.gather(1, kk[:, None])[:, 0]
            else:
                j = sc.argmax(-1)
                op_i, tl_i = j // 100, j % 100
                rl = self.raw_logits(zf[:, None], ht, op_i[:, None], tl_i[:, None], up[:, i:i + 1])[:, 0]
                if raw_mask is not None:
                    rl = raw_mask(i, op_i, tl_i, rl)
                rw_i = rl.argmax(-1)
            il, ql = self.arg_logits(zf[:, None], ht, rw_i[:, None], tl_i[:, None])
            has_it = (rw_i == 5) | (rw_i == 6) | (rw_i == 13)
            OP[:, i], TL[:, i], RW[:, i] = op_i, tl_i, rw_i
            IT[:, i], QT[:, i] = torch.where(has_it, il[:, 0].argmax(-1), 0), ql[:, 0].argmax(-1)
        return dict(op=OP, tile=TL, raw=RW, item=IT, qty=QT), self.market_heads(h)
