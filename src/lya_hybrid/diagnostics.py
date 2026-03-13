from __future__ import annotations

import numpy as np

from .projection_1d import Polynomial1DCounterterms, project_to_1d


def projection_convergence_scan(
    *,
    kpar_values: np.ndarray,
    p3d_callable,
    kmax_proj: float,
    nint_values: list[int],
    counterterms: Polynomial1DCounterterms | None = None,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for nint in nint_values:
        out[nint] = project_to_1d(
            kpar_values=kpar_values,
            p3d_callable=p3d_callable,
            kmax_proj=kmax_proj,
            nint=nint,
            method="trapz",
            counterterms=counterterms,
        )["total"]
    return out
