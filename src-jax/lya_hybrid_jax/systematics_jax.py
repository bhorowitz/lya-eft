"""JAX implementations of multiplicative systematics factors for Lyman-alpha P1D.

Mirrors the systematics functions in run_2405_stage1_sdss_baseline.py but as
importable, JIT-compiled JAX functions.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import jit


@jit
def _hmpc_to_kms_jax(z, h, omega_b, omega_cdm):
    omega_m = omega_b + omega_cdm
    ez = jnp.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))
    return h * (1.0 + z) / (100.0 * h * ez)


@jit
def paper_systematics_factor_jax(z, kpar_hmpc, h, omega_b, omega_cdm):
    """JIT-compiled multiplicative SiIII + thermal-broadening factor (JAX scalars/arrays)."""
    kpar_kms = _hmpc_to_kms_jax(z, h, omega_b, omega_cdm) * kpar_hmpc
    fbar     = jnp.exp(-0.0025 * (1.0 + z) ** 3.7)
    srat     = 8.7e-3 / jnp.maximum(1.0 - fbar, 1.0e-8)
    ksi      = 1.0 + 2.0 * srat * jnp.cos(2.0 * jnp.pi / 0.0028 * kpar_kms) + srat ** 2
    return ksi * jnp.exp(-((kpar_kms / 0.11) ** 2))


def paper_systematics_factor_jnp(z, kpar_hmpc, h, omega_b, omega_cdm):
    """Non-JIT version — accepts mixed JAX/numpy; for use inside other @jit functions."""
    om  = omega_b + omega_cdm
    ez  = jnp.sqrt(om * (1.0 + z) ** 3 + (1.0 - om))
    fac = h * (1.0 + z) / (100.0 * h * ez)
    kpar_kms = fac * kpar_hmpc
    fbar = jnp.exp(-0.0025 * (1.0 + z) ** 3.7)
    srat = 8.7e-3 / jnp.maximum(1.0 - fbar, 1.0e-8)
    ksi  = 1.0 + 2.0 * srat * jnp.cos(2.0 * jnp.pi / 0.0028 * kpar_kms) + srat ** 2
    return ksi * jnp.exp(-((kpar_kms / 0.11) ** 2))
