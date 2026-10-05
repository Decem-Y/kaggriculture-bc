"""BC training, used for all three stages (see configs/).

  python train.py --config configs/pretrain.yaml --run pretrain
  python train.py --config configs/finetune.yaml --run finetune
  python train.py --config configs/specialize.yaml --run specialize

Checkpoints hold the EMA weights; they go to runs/<run>/ (stepN.pt every ckpt_every, best.pt = lowest validation loss).
"""
import argparse, datetime, json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import build_index, loader, fixed_val  # noqa: E402
from model import BCNet  # noqa: E402


def to_dev(d, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in d.items()}


def ce(logits, target, mask=None):
    l = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    if mask is None:
        return l.mean()
    m = mask.reshape(-1).float()
    return (l * m).sum() / m.sum().clamp(min=1)


def unit_ok(raw_p, item_p, qty_p, y):
    """Exact raw-action match: same op; same item for PICKUP / PLANT / PLACE; same quantity bin for PICKUP / PLACE."""
    rop = y["r_op"]
    ok = raw_p == rop
    has_it = (rop == 5) | (rop == 6) | (rop == 13)
    has_q = (rop == 5) | (rop == 13)
    return ok & (~has_it | (item_p == y["r_item"])) & (~has_q | (qty_p == y["r_qty"]))


def compute(model, x, y, cfg):
    uop = y["u_op"]
    rop = y["r_op"]
    valid = uop >= 0
    op_l, tile_l, raw_l, item_l, qty_l, mk = model(x, uop, y["u_tile"], rop, y["r_item"])
    has_item = valid & ((rop == 5) | (rop == 6) | (rop == 13))
    has_qty = valid & ((rop == 5) | (rop == 13))
    L = dict(op=ce(op_l, uop.clamp(min=0), valid), tile=ce(tile_l, y["u_tile"], valid), raw=ce(raw_l, rop.clamp(min=0), valid),
             item=ce(item_l, y["r_item"], has_item), qty=ce(qty_l, y["r_qty"], has_qty),
             sell=ce(mk["sell"], y["m_sell"]), seed=ce(mk["seed"], y["m_seed"]), buyp=ce(mk["buyp"], y["m_buyp"]),
             anim=ce(mk["anim"], y["m_anim"]), hire=ce(mk["hire"], y["m_hire"]), land=ce(mk["land"], y["m_land"]))
    out = dict(op_l=op_l, tile_l=tile_l, raw_l=raw_l, item_l=item_l, qty_l=qty_l, mk=mk, valid=valid, has_item=has_item, has_qty=has_qty)
    w = cfg["loss_w"]
    total = sum(w.get(k, 1.0) * v for k, v in L.items())
    return total, L, out


@torch.no_grad()
def evaluate(model, vx, vy, dev, bs=512, ar=True):
    """Per-head losses and accuracies. *_tf use teacher forcing; *_ar decode freely (what the agent executes)."""
    model.eval()
    n = len(vy["u_op"])
    acc, cnt, losses = {}, {}, {}

    def add(k, correct, mask):
        acc[k] = acc.get(k, 0) + float((correct & mask).sum())
        cnt[k] = cnt.get(k, 0) + float(mask.sum())

    for i in range(0, n, bs):
        x = to_dev({k: torch.from_numpy(v[i:i + bs]) for k, v in vx.items()}, dev)
        y = to_dev({k: torch.from_numpy(v[i:i + bs]) for k, v in vy.items()}, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, L, o = compute(model, x, y, {"loss_w": {}})
        for k, v in L.items():
            losses[k] = losses.get(k, 0) + float(v) * len(y["u_op"])
        v = o["valid"]
        opp = o["op_l"].argmax(-1)
        tp = o["tile_l"].argmax(-1)
        rp = o["raw_l"].argmax(-1)
        rop = y["r_op"]
        add("op", opp == y["u_op"], v)
        add("tile|op", tp == y["u_tile"], v)
        add("joint", (opp == y["u_op"]) & (tp == y["u_tile"]), v)
        add("raw", rp == rop, v)
        add("raw_move", rp == rop, v & (rop >= 1) & (rop <= 4))
        add("raw_task", rp == rop, v & (rop >= 5))
        add("item", o["item_l"].argmax(-1) == y["r_item"], o["has_item"])
        add("qty", o["qty_l"].argmax(-1) == y["r_qty"], o["has_qty"])
        ok = unit_ok(rp, o["item_l"].argmax(-1), o["qty_l"].argmax(-1), y)
        add("unit_tf", ok, v)
        add("step_tf", (ok | ~v).all(-1), torch.ones_like(v[:, 0]))
        if ar:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dd, _ = model.decode(x)
            ok = unit_ok(dd["raw"], dd["item"], dd["qty"], y)
            add("unit_ar", ok, v)
            add("step_ar", (ok | ~v).all(-1), torch.ones_like(v[:, 0]))
            add("farmer_ar", ok[:, 0], v[:, 0])
        for k in ["sell", "seed", "buyp", "anim", "hire", "land"]:
            p = o["mk"][k].argmax(-1)
            add(k, p == y["m_" + k], torch.ones_like(p, dtype=torch.bool))
            add(k + "_nz", p == y["m_" + k], y["m_" + k] > 0)
    model.train()
    res = {k: acc[k] / max(cnt[k], 1) for k in acc}
    res.update({"loss_" + k: v / n for k, v in losses.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--smoke", action="store_true", help="short run with the config's `smoke` overrides")
    ap.add_argument("--resume", default=None, help="continue from runs/<run>/last.pt (with optimizer state)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, a.config)))
    out = os.path.join(ROOT, "runs", a.run)
    os.makedirs(out, exist_ok=True)
    if a.smoke:
        cfg.update(cfg.get("smoke", {}))
    if a.resume:
        cfg["resume"] = a.resume
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.steps is not None:
        cfg["steps"] = a.steps
    yaml.safe_dump(cfg, open(os.path.join(out, "config.resolved.yaml"), "w"), allow_unicode=True)
    logf = open(os.path.join(out, "train.log"), "a")

    def log(*s):
        msg = " ".join(str(v) for v in s)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.manual_seed(cfg["seed"])
    if cfg.get("init"):
        vocab = torch.load(os.path.join(ROOT, cfg["init"]), map_location="cpu", weights_only=False)["vocab"]
    else:
        vocab = json.load(open(os.path.join(ROOT, cfg["vocab_file"])))
    eps = build_index(os.path.join(ROOT, cfg["census"]), os.path.join(ROOT, cfg["ep_dir"]), os.path.join(ROOT, cfg["lab_dir"]),
                      vocab, since=cfg["since"], val_mod=cfg.get("val_mod", 50), ndate=cfg["model"].get("ndate"))
    json.dump(vocab, open(os.path.join(out, "vocab.json"), "w"), ensure_ascii=False)
    tr = [e for e in eps if not e["val"]]
    va = sorted([e for e in eps if e["val"]], key=lambda e: e["day"], reverse=True)   # validate on the most recent games
    sf = None
    if cfg.get("ft_teams"):                    # specialisation: train / validate only on these teams' seats
        ids = {vocab.index(t) for t in cfg["ft_teams"] if t in vocab}
        assert ids, f"none of {cfg['ft_teams']} is in the vocabulary"
        sf = lambda e, s: e["team"][s] in ids
        tr = [e for e in tr if sf(e, 0) or sf(e, 1)]
        va = [e for e in va if sf(e, 0) or sf(e, 1)]
    log(f"episodes train {len(tr)} val {len(va)} teams {len(vocab)}")
    dev = torch.device("cuda")
    model = BCNet(len(vocab), **cfg["model"]).to(dev)
    if cfg.get("init"):
        ck0 = torch.load(os.path.join(ROOT, cfg["init"]), map_location="cpu", weights_only=False)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck0["model"].items()})
        log("init from", cfg["init"], "step", ck0["step"])
    log("params", sum(p.numel() for p in model.parameters()))
    vx, vy = fixed_val(va, n_eps=cfg["val_eps"], per_ep=cfg["val_per_ep"], seat_filter=sf)
    log("val samples", len(vy["u_op"]))
    tvx = None
    if cfg.get("teacher") and sf is None:      # extra validation on one team's seats only
        tid = vocab.index(cfg["teacher"])
        tsf = lambda e, s: e["team"][s] == tid
        va_t = [e for e in va if any(e["mask"][s] and tsf(e, s) for s in range(2))]
        tvx, tvy = fixed_val(va_t, n_eps=len(va_t), per_ep=cfg["val_per_ep"], seat_filter=tsf)
        log(f"val {cfg['teacher']}: episodes {len(va_t)} samples {len(tvy['u_op'])}")
    wts = None
    if cfg.get("recency_halflife"):            # sampling weight halves every recency_halflife days back from the newest game
        dd = [datetime.date.fromisoformat(e["day"]) for e in tr]
        last = max(dd)
        wts = [0.5 ** ((last - d).days / float(cfg["recency_halflife"])) for d in dd]
        W = np.array(wts)
        log(f"recency half-life {cfg['recency_halflife']} d: effective episodes {W.sum() ** 2 / (W ** 2).sum():.0f} of {len(W)}")
    dl = loader(tr, cfg["batch_size"], workers=cfg["workers"], pool=cfg["pool"], seed=cfg["seed"], seat_filter=sf, weights=wts)
    if cfg.get("optim", "adamw") == "muon":
        from optim import build_optimizer
        opt = build_optimizer(model, cfg, log)
    else:
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "e_" not in n.split(".")[0]]
        nodecay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "e_" not in n.split(".")[0])]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["wd"]}, {"params": nodecay, "weight_decay": 0.0}],
                                lr=cfg["lr"], betas=(0.9, 0.95), fused=True)
    total = cfg["steps"]
    warm = cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(        # linear warmup, cosine decay to 5%
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))))
    cmodel = torch.compile(model) if cfg.get("compile") else model
    if cfg.get("resume"):
        ckr = torch.load(os.path.join(ROOT, cfg["resume"]), map_location="cpu", weights_only=False)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckr.get("model_raw", ckr["model"]).items()})
        opt.load_state_dict(ckr["opt"])
        for _ in range(ckr["step"]):
            sched.step()
        log("resumed from", cfg["resume"], "step", ckr["step"])
    # Only EMA weights are evaluated and saved: raw weights at high learning rate play far worse in closed loop.
    ema_decay = float(cfg.get("ema", 0.0))
    ema = None
    if ema_decay > 0:
        ema = BCNet(len(vocab), **cfg["model"]).to(dev)
        src = ckr["model"] if (cfg.get("resume") and "model_raw" in ckr) else model.state_dict()
        ema.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in src.items()})
        ema.requires_grad_(False)
        ema_p, mod_p = list(ema.parameters()), list(model.parameters())
        n_ema = int(ckr["step"]) if cfg.get("resume") else 0
    step = step0 = int(ckr["step"]) if cfg.get("resume") else 0
    t0, run = time.time(), {}
    run_tot, nbad, nnan = 1e9, 0, 0
    gn_ref, nspike = None, 0
    best = 1e9
    for x, y in dl:
        x, y = to_dev(x, dev), to_dev(y, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, L, _ = compute(cmodel, x, y, cfg)
        lv = float(loss.detach())
        if step - step0 > 300 and lv > 5 * run_tot:      # outlier batch: skip
            nbad += 1
            log(f"BAD batch at step {step}: loss {lv:.2f} vs running {run_tot:.3f}")
            if nbad > 200:
                raise RuntimeError("too many bad batches")
            continue
        run_tot = run_tot * 0.99 + lv * 0.01 if run_tot < 1e8 else lv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        gv = float(gn)
        # Skip steps whose gradient norm is far above a clipped running reference (or above 1e3). Clipping alone
        # keeps a garbage direction, and Muon turns a rank-one spike in its momentum into a full-size update.
        if math.isfinite(gv) and (gv > 1e3 or (gn_ref is not None and step - step0 > 100 and gv > 20 * gn_ref)):
            nspike += 1
            log(f"GNSPIKE at step {step}: gn {gv:.3g} vs ref {gn_ref if gn_ref is not None else float('nan'):.3g}"
                + ("" if cfg.get("skip_gnspike") else " (not skipped)"))
            if cfg.get("skip_gnspike"):
                opt.zero_grad(set_to_none=True)
                if nspike > 600:
                    raise RuntimeError("too many gradient-norm spikes")
                continue
        elif math.isfinite(gv):
            gn_ref = gv if gn_ref is None else 0.99 * gn_ref + 0.01 * min(gv, 3 * gn_ref)
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            nnan += 1
            log(f"NONFINITE at step {step}: loss {float(loss)} gn {float(gn)}")
            opt.zero_grad(set_to_none=True)
            if nnan > 600:
                raise RuntimeError("too many non-finite steps")
            continue
        opt.step()
        sched.step()
        step += 1
        if ema is not None:
            n_ema += 1
            with torch.no_grad():
                torch._foreach_lerp_(ema_p, mod_p, 1.0 - min(ema_decay, (1 + n_ema) / (10 + n_ema)))
        for k, v in L.items():
            v = float(v.detach())
            run[k] = run.get(k, 0) * 0.98 + v * 0.02 if step > 1 else v
        if step % cfg["log_every"] == 0:
            el = time.time() - t0
            log(f"step {step} lr {sched.get_last_lr()[-1]:.2e} loss {float(loss):.4f} gn {float(gn):.2f} "
                + " ".join(f"{k}={v:.3f}" for k, v in run.items()) + f" | {(step - step0) * cfg['batch_size'] / el:.0f} samp/s")
        if step % cfg["eval_every"] == 0 or step == total:
            evm = ema if ema is not None else model
            r = evaluate(evm, vx, vy, dev)
            log(f"EVAL step {step} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(r.items())))
            if tvx is not None:
                rt = evaluate(evm, tvx, tvy, dev)
                log(f"EVAL_TEACHER step {step} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(rt.items()) if not k.startswith("loss_")))
                r.update({"teacher_" + k: v for k, v in rt.items()})
            vl = sum(v for k, v in r.items() if k.startswith("loss_"))
            ck = dict(model=evm.state_dict(), vocab=vocab, cfg=cfg, step=step, eval=r)
            torch.save(dict(ck, opt=opt.state_dict(), model_raw=model.state_dict()), os.path.join(out, "last.pt"))
            if cfg.get("ckpt_every") and step % cfg["ckpt_every"] == 0:
                torch.save(ck, os.path.join(out, f"step{step}.pt"))
            if vl < best:
                best = vl
                torch.save(ck, os.path.join(out, "best.pt"))
            json.dump(dict(step=step, best_val_loss=best, eval=r), open(os.path.join(out, "summary.json"), "w"), indent=1)
        if step >= total:
            break
    log("done", f"{time.time() - t0:.0f}s", f"nonfinite {nnan} bad {nbad} gnspike {nspike}")


if __name__ == "__main__":
    main()
