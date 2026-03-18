"""JAX/GPU P1D projection helpers.

Mirrors lya_hybrid.projection_1d.project_to_1d (trapz method) but runs
entirely on GPU via vmap + JIT.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit


# ---------------------------------------------------------------------------
# Bilinear interpolation on a regular (k, mu) grid
# ---------------------------------------------------------------------------
@jit
def _bilinear_interp_jax(
    k_g: jnp.ndarray,   # (nk_g,)
    mu_g: jnp.ndarray,  # (nmu_g,)
    p3d_g: jnp.ndarray, # (nk_g, nmu_g)
    k_q: jnp.ndarray,   # (n,)
    mu_q: jnp.ndarray,  # (n,)
) -> jnp.ndarray:
    """Bilinear interpolation — matches scipy RegularGridInterpolator(method='linear')."""
    nk, nmu = k_g.shape[0], mu_g.shape[0]
    kc  = jnp.clip(k_q,  k_g[0],  k_g[-1])
    mc  = jnp.clip(mu_q, mu_g[0], mu_g[-1])
    ik  = jnp.clip(jnp.searchsorted(k_g,  kc, side="right") - 1, 0, nk  - 2)
    im  = jnp.clip(jnp.searchsorted(mu_g, mc, side="right") - 1, 0, nmu - 2)
    wk  = (kc - k_g[ik])  / (k_g[ik + 1]  - k_g[ik])
    wm  = (mc - mu_g[im]) / (mu_g[im + 1] - mu_g[im])
    return (
        (1 - wk) * (1 - wm) * p3d_g[ik, im]
        +      wk * (1 - wm) * p3d_g[ik + 1, im]
        + (1 - wk) *      wm * p3d_g[ik, im + 1]
        +      wk *       wm * p3d_g[ik + 1, im + 1]
    )


# ---------------------------------------------------------------------------
# P1D projection
# ---------------------------------------------------------------------------
@partial(jit, static_argnames=("nint", "kmax_proj"))
def project_to_1d_jax(
    kpar_values: jnp.ndarray,  # (nkpar,)
    k_g: jnp.ndarray,          # (nk_g,)
    mu_g: jnp.ndarray,         # (nmu_g,)
    p3d_g: jnp.ndarray,        # (nk_g, nmu_g)
    kmax_proj: float,
    nint: int,
) -> jnp.ndarray:
    """
    Vectorised JAX projection:  P1D(kpar) = (1/2π) ∫_{kpar}^{kmax} k P3D(k, kpar/k) dk

    Mirrors lya_hybrid.projection_1d.project_to_1d (trapz method).
    """
    kmax_f = float(kmax_proj)

    def one(kpar):
        k_v  = jnp.exp(jnp.linspace(jnp.log(kpar), jnp.log(kmax_f), nint))
        mu_v = jnp.clip(kpar / k_v, 0.0, 1.0)
        p3d  = _bilinear_interp_jax(k_g, mu_g, p3d_g, k_v, mu_v)
        return jnp.trapezoid(k_v * p3d, k_v) / (2.0 * jnp.pi)

    return jax.vmap(one)(kpar_values)


# ---------------------------------------------------------------------------
# P3D grid container — pre-allocated once per z-bin
# ---------------------------------------------------------------------------
@dataclass
class JaxP3DGrid:
    """Pre-computed (k, mu) meshgrid arrays used for the P3D→P1D pipeline."""
    k_g:     jnp.ndarray   # (nk_g,)
    mu_g:    jnp.ndarray   # (nmu_g,)
    kk_flat: jnp.ndarray   # (nk_g * nmu_g,) — flattened meshgrid k values
    mm_flat: jnp.ndarray   # (nk_g * nmu_g,) — flattened meshgrid mu values
    kpar:    jnp.ndarray   # (nkpar,)
    nk_g:    int
    nmu_g:   int


def make_jax_p3d_grid(
    kpar_hmpc: np.ndarray,
    kmax_proj: float,
    nk_g: int = 40,
    nmu_g: int = 16,
) -> JaxP3DGrid:
    """Build the (k, mu) evaluation grid used by project_to_1d_jax."""
    k_g  = np.logspace(
        np.log10(max(float(np.min(kpar_hmpc)) * 0.95, 1.0e-4)),
        np.log10(kmax_proj),
        nk_g,
    )
    mu_g = np.linspace(0.0, 1.0, nmu_g)
    kk, mm = np.meshgrid(k_g, mu_g, indexing="ij")
    return JaxP3DGrid(
        k_g=jnp.array(k_g),
        mu_g=jnp.array(mu_g),
        kk_flat=jnp.array(kk.ravel()),
        mm_flat=jnp.array(mm.ravel()),
        kpar=jnp.array(kpar_hmpc),
        nk_g=nk_g,
        nmu_g=nmu_g,
    )
