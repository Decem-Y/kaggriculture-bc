"""Single-device Muon with an auxiliary AdamW, adapted from Keller Jordan's reference implementation
(https://github.com/KellerJordan/Muon, MIT License).

Changes relative to the reference:
- single-device only;
- parameters with ndim > 2 are flattened to (out, -1) before orthogonalisation;
- parameters whose grad is None are skipped (no zero-grad injection);
- `chunks`: stacked projections (e.g. a fused QKV weight) are split into blocks and orthogonalised separately,
  so one block cannot dominate the shared spectral scale;
- `ortho`: optional (B, M, N) -> (B, M, N) callable replacing the built-in Newton-Schulz iteration;
- `batch_shapes`: matrices with the same (rows, cols) after flattening / splitting are orthogonalised in one call.
"""
import torch


def zeropower_via_newtonschulz5(G, steps: int):
    """Quintic Newton-Schulz iteration approximating the orthogonal factor of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _pre_orthogonalize(grad, momentum, beta, nesterov, chunks):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim > 2:
        update = update.view(len(update), -1)
    if chunks > 1:
        rows, cols = update.shape
        assert rows % chunks == 0, f"{rows} rows cannot be split into {chunks} chunks"
        update = update.view(chunks, rows // chunks, cols)
    return update


def _orthogonalize(update, ns_steps, ortho):
    return zeropower_via_newtonschulz5(update, steps=ns_steps) if ortho is None else ortho(update)


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, chunks=1, ortho=None):
    update = _pre_orthogonalize(grad, momentum, beta, nesterov, chunks)
    update = _orthogonalize(update, ns_steps, ortho)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update


def _muon_step_batched(group, get_state, ortho):
    buckets = {}
    for p in group["params"]:
        if p.grad is None:
            continue
        state = get_state(p)
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(p)
        u = _pre_orthogonalize(p.grad, state["momentum_buffer"], group["momentum"], True, group["chunks"])
        r, c = u.shape[-2], u.shape[-1]
        buckets.setdefault((r, c), []).append((p, u.reshape(-1, r, c)))

    for (r, c), items in buckets.items():
        stacked = torch.cat([u for _, u in items], dim=0) if len(items) > 1 else items[0][1]
        stacked = _orthogonalize(stacked, group["ns_steps"], ortho)
        stacked *= max(1, r / c)**0.5
        off = 0
        for p, u in items:
            k = u.size(0)
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(stacked[off:off + k].reshape(p.shape), alpha=-group["lr"])
            off += k


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Every param group carries `use_muon`. Muon groups take lr / momentum / weight_decay / ns_steps / chunks;
    AdamW groups take lr / betas / eps / weight_decay. A LambdaLR scales all groups by the same factor."""

    def __init__(self, param_groups, ortho=None, batch_shapes=False):
        self._ortho, self._batch_shapes = ortho, batch_shapes
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["chunks"] = group.get("chunks", 1)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "ns_steps", "chunks", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                if self._batch_shapes:
                    _muon_step_batched(group, lambda p: self.state[p], self._ortho)
                    continue
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"],
                                         ns_steps=group["ns_steps"], chunks=group["chunks"], ortho=self._ortho)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
