"""Torch checkpoint -> numpy weights (weights.npz) + team vocabulary (vocab.json)."""
import json, os

import numpy as np
import torch


def export(ckpt_path, out_dir):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    os.makedirs(out_dir, exist_ok=True)
    sd = {k.replace("_orig_mod.", ""): v.float().numpy() for k, v in ck["model"].items()}
    m = ck["cfg"]["model"]
    sd["_layers"] = np.array(m["layers"])
    sd["_heads"] = np.array(m["heads"])
    sd["_cap"] = np.array(float(m.get("cap", 0.0)))
    sd["_rope"] = np.array(float(m.get("rope_theta", 0.0)))
    np.savez(os.path.join(out_dir, "weights.npz"), **sd)
    json.dump(ck["vocab"], open(os.path.join(out_dir, "vocab.json"), "w"), ensure_ascii=False)
    return ck
