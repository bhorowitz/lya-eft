from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate


@dataclass
class Polynomial1DCounterterms:
    c0: float = 0.0
    c2: float = 0.0
    c4: float = 0.0

    def evaluate(self, kpar: np.ndarray) -> np.ndarray:
        return self.c0 + self.c2 * kpar**2 + self.c4 * kpar**4


def project_to_1d(
    *,
    kpar_values: np.ndarray,
    p3d_callable: Callable[[np.ndarray, np.ndarray], np.ndarray],
    kmax_proj: float,
    nint: int = 1200,
    method: str = "trapz",
    counterterms: Polynomial1DCounterterms | None = None,
) -> dict[str, np.ndarray]:
    kpar_values = np.asarray(kpar_values)
    raw = np.zeros_like(kpar_values)

    for i, kpar in enumerate(kpar_values):
        if kpar >= kmax_proj:
            raw[i] = 0.0
            continue

        if method == "trapz":
            kvals = np.logspace(np.log10(kpar), np.log10(kmax_proj), nint)
            muvals = np.clip(kpar / kvals, 0.0, 1.0)
            integrand = kvals * p3d_callable(kvals, muvals)
            raw[i] = integrate.trapezoid(integrand, kvals) / (2.0 * np.pi)
        elif method == "quad":
            def _integrand(k: float) -> float:
                mu = float(np.clip(kpar / k, 0.0, 1.0))
                return k * float(p3d_callable(np.array([k]), np.array([mu]))[0])

            raw[i] = integrate.quad(_integrand, kpar, kmax_proj, epsabs=1.0e-8, epsrel=1.0e-5, limit=300)[0] / (
                2.0 * np.pi
            )
        else:
            raise ValueError(f"Unknown projection method: {method}")

    counter = np.zeros_like(raw)
    if counterterms is not None:
        counter = counterterms.evaluate(kpar_values)

    return {
        "raw": raw,
        "counterterms": counter,
        "total": raw + counter,
    }
