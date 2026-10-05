"""Standard Newton-Schulz orthogonalisation with configurable coefficients, used as Muon's `ortho` callable.

Trimmed from https://github.com/Dao-AILab/gram-newton-schulz (MIT License): only the plain-PyTorch standard
iteration is kept, plus an `ns_dtype` argument. Polar Express coefficients: https://arxiv.org/abs/2505.16932.
"""
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
from torch import Tensor

YOU_COEFFICIENTS = [
    [4.0848, -6.8946, 2.9270],
    [3.9505, -6.3029, 2.6377],
    [3.7418, -5.5913, 2.3037],
    [2.8769, -3.1427, 1.2046],
    [2.8366, -3.0525, 1.2012],
]

_unmodified_polar_express_coefficients = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
safety_factor = 1.05
POLAR_EXPRESS_COEFFICIENTS = [
    (a / safety_factor, b / safety_factor**3, c / safety_factor**5)
    for (a, b, c) in _unmodified_polar_express_coefficients
]

KELLER_COEFFICIENTS = [(3.4445, -4.7750, 2.0315)] * 5

COEFFICIENT_PRESETS = {
    "polar_express": POLAR_EXPRESS_COEFFICIENTS,
    "you": YOU_COEFFICIENTS,
    "keller": KELLER_COEFFICIENTS,
}

_OPS = SimpleNamespace(
    sym_mm=lambda A, B: A @ B,
    sym_baddbmm=lambda A, B, C, alpha=1., beta=1.: torch.baddbmm(C, A, B, alpha=alpha, beta=beta),
    mm_add=lambda A, B, C, beta: torch.baddbmm(C, A, B, beta=beta),
)


def _make_standard(ns_coefficients, ns_epsilon, compile_kwargs, ns_dtype):
    ns_coefficients = list(ns_coefficients)
    ops = _OPS

    def _standard_newton_schulz(X: Tensor) -> Tensor:
        tall_skinny = X.size(-2) > X.size(-1)
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + ns_epsilon)
        X = X.to(ns_dtype)
        for a, b, c in ns_coefficients:
            if tall_skinny:
                A = ops.sym_mm(X.mT, X)
            else:
                A = ops.sym_mm(X, X.mT)
            B = ops.sym_baddbmm(A, A, C=A, alpha=c, beta=b)
            if tall_skinny:
                X = ops.mm_add(X, B, C=X, beta=a)
            else:
                X = ops.mm_add(B, X, C=X, beta=a)
        return X

    if compile_kwargs is not None:
        _standard_newton_schulz = torch.compile(_standard_newton_schulz, **compile_kwargs)
    return _standard_newton_schulz


class StandardNewtonSchulz:
    """X: (M, N) or (B, M, N) -> approximately orthogonalised X, same shape and dtype."""

    def __init__(self, ns_epsilon: float = 1e-7, ns_coefficients: Optional[List[List[float]]] = None,
                 compile_kwargs: Optional[Dict] = None, ns_dtype: torch.dtype = torch.float16):
        if compile_kwargs is not None:
            # one compiled graph per matrix shape; the default cache limit (8) is too small
            torch._dynamo.config.cache_size_limit = max(128, torch._dynamo.config.cache_size_limit)
        coefs = ns_coefficients if ns_coefficients is not None else POLAR_EXPRESS_COEFFICIENTS
        self._fn = _make_standard(coefs, ns_epsilon, compile_kwargs, ns_dtype)

    def __call__(self, X: Tensor) -> Tensor:
        original_shape = X.shape
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            X = X.view(-1, *X.shape[-2:])
        original_dtype = X.dtype
        return self._fn(X).to(original_dtype).view(original_shape)
