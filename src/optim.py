"""Muon for hidden 2-D weights, AdamW for everything else.

Grouping rules:
1. ndim <= 1 (LayerNorm, biases, the null memory slot) -> AdamW.
2. Embedding tables -> AdamW.
3. Output heads (parent module name ends with "head", plus raw_out) -> AdamW.
4. Fused encoder QKV weights are orthogonalised as 3 blocks, fused decoder KV weights as 2 blocks.
5. All other 2-D weights -> Muon.
Within AdamW, weight decay applies to tensors with dim >= 2 whose top-level name does not start with "e_".
"""
import torch.nn as nn
import torch

from newton_schulz import COEFFICIENT_PRESETS, StandardNewtonSchulz
from muon import SingleDeviceMuonWithAuxAdam

OUT_HEADS = {"raw_out"}


def _is_output_head(parts):
    return len(parts) >= 2 and (parts[-2].endswith("head") or parts[-2] in OUT_HEADS)


def split_params(model, split_qkv=True):
    """-> (muon, qkv, kv, adam_decay, adam_no_decay)."""
    emb = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    muon, qkv, kv, adam_d, adam_n = [], [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        leaf = parts[-1]
        to_adam = (p.ndim <= 1 or id(p) in emb or not (leaf.endswith("weight") or leaf.endswith("bias"))
                   or _is_output_head(parts))
        if to_adam:
            (adam_d if (p.dim() >= 2 and "e_" not in parts[0]) else adam_n).append(p)
        elif split_qkv and name.endswith(".qkv.weight") and p.shape[0] == 3 * p.shape[1]:
            qkv.append(p)
        elif split_qkv and name.endswith(".kv.weight") and p.shape[0] == 2 * p.shape[1]:
            kv.append(p)
        else:
            muon.append(p)
    return muon, qkv, kv, adam_d, adam_n


def build_optimizer(model, cfg, log=print):
    mo = cfg["muon"]
    muon, qkv, kv, adam_d, adam_n = split_params(model, mo.get("split_qkv", True))
    assert muon, "empty Muon group"
    n = lambda ps: sum(p.numel() for p in ps)
    log(f"[optim] Muon {n(muon) / 1e6:.2f}M ({len(muon)}) + QKV {n(qkv) / 1e6:.2f}M ({len(qkv)}) + KV {n(kv) / 1e6:.2f}M ({len(kv)}) "
        f"| AdamW {n(adam_d) / 1e6:.2f}M decay ({len(adam_d)}) + {n(adam_n) / 1e6:.2f}M no decay ({len(adam_n)})")
    mk = lambda ps, ch: dict(params=ps, use_muon=True, lr=mo["lr"], momentum=mo["momentum"], weight_decay=mo["wd"],
                             ns_steps=mo["ns_steps"], chunks=ch)
    groups = [mk(muon, 1)] + ([mk(qkv, 3)] if qkv else []) + ([mk(kv, 2)] if kv else []) + [
        dict(params=adam_d, use_muon=False, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg["wd"]),
        dict(params=adam_n, use_muon=False, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)]
    coefs = COEFFICIENT_PRESETS[mo["ns_coefs"]]
    assert len(coefs) == mo["ns_steps"], "ns_coefs and ns_steps disagree"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[mo["ns_dtype"]]
    ortho = StandardNewtonSchulz(ns_coefficients=coefs, ns_dtype=dtype,
                                 compile_kwargs={"fullgraph": True, "mode": "reduce-overhead"} if mo["ns_compile"] else None)
    return SingleDeviceMuonWithAuxAdam(groups, ortho=ortho, batch_shapes=mo["ns_batch"])
