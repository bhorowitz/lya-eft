"""JAX/GPU port of IvanovFullModel — loop integrals on GPU via JIT.

Mirrors src/lya_hybrid/model_ivanov_full.py but with JAX operations so that
the loop quadrature runs on GPU.  The quadrature grid is constructed once in
__init__ (numpy); every call to evaluate_components dispatches to the JIT-
compiled _loop_terms_jax kernel.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from lya_hybrid.model_ivanov_full import IvanovFullParams


# ---------------------------------------------------------------------------
# Params container — NamedTuple so JAX treats it as a pytree automatically
# ---------------------------------------------------------------------------
class _IvanovParamsJAX(NamedTuple):
    """Mirrors IvanovFullParams as a JAX-traceable pytree."""
    b1: float; b_eta: float
    b_delta2: float; b_G2: float; b_KK_par: float
    b_delta_eta: float; b_eta2: float; b_Pi2_par: float
    b_gamma3: float; b_delta_Pi2_par: float; b_eta_Pi2_par: float
    b_KPi2_par: float; b_Pi3_par: float
    c0_ct: float; c2_ct: float; c4_ct: float


def _to_jax_params(p: IvanovFullParams) -> _IvanovParamsJAX:
    return _IvanovParamsJAX(
        b1=p.b1, b_eta=p.b_eta,
        b_delta2=p.b_delta2, b_G2=p.b_G2, b_KK_par=p.b_KK_par,
        b_delta_eta=p.b_delta_eta, b_eta2=p.b_eta2, b_Pi2_par=p.b_Pi2_par,
        b_gamma3=getattr(p, "b_gamma3", 0.0),
        b_delta_Pi2_par=p.b_delta_Pi2_par, b_eta_Pi2_par=p.b_eta_Pi2_par,
        b_KPi2_par=p.b_KPi2_par, b_Pi3_par=p.b_Pi3_par,
        c0_ct=p.c0_ct, c2_ct=p.c2_ct, c4_ct=p.c4_ct,
    )


# ---------------------------------------------------------------------------
# SPT kernel helpers  (all JIT-compiled)
# ---------------------------------------------------------------------------
@jit
def _safe_div(num, den, eps=1.0e-20):
    num_b, den_b = jnp.broadcast_arrays(num, den)
    return jnp.where(jnp.abs(den_b) > eps, num_b / den_b, 0.0)


@jit
def _add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


@jit
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@jit
def _norm2(a):
    return _dot(a, a)


@jit
def _alpha(a, b):
    return _safe_div(_dot(_add(a, b), a), _norm2(a))


@jit
def _beta(a, b):
    ab = _add(a, b)
    return _safe_div(_norm2(ab) * _dot(a, b), 2.0 * _norm2(a) * _norm2(b))


@jit
def _f2_unsym(a, b):
    return (5.0 / 7.0) * _alpha(a, b) + (2.0 / 7.0) * _beta(a, b)


@jit
def _g2_unsym(a, b):
    return (3.0 / 7.0) * _alpha(a, b) + (4.0 / 7.0) * _beta(a, b)


@jit
def _f2_sym(a, b):
    return 0.5 * (_f2_unsym(a, b) + _f2_unsym(b, a))


@jit
def _g2_sym(a, b):
    return 0.5 * (_g2_unsym(a, b) + _g2_unsym(b, a))


@jit
def _f3_unsym(a, b, c):
    bc, ab = _add(b, c), _add(a, b)
    return (
        7.0 * _alpha(a, bc) * _f2_unsym(b, c) + 2.0 * _beta(a, bc) * _g2_unsym(b, c)
        + _g2_unsym(a, b) * (7.0 * _alpha(ab, c) + 2.0 * _beta(ab, c))
    ) / 18.0


@jit
def _g3_unsym(a, b, c):
    bc, ab = _add(b, c), _add(a, b)
    return (
        3.0 * _alpha(a, bc) * _f2_unsym(b, c) + 6.0 * _beta(a, bc) * _g2_unsym(b, c)
        + _g2_unsym(a, b) * (3.0 * _alpha(ab, c) + 6.0 * _beta(ab, c))
    ) / 18.0


@jit
def _f3_sym(a, b, c):
    return sum(
        _f3_unsym(*perm) for perm in
        ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a))
    ) / 6.0


@jit
def _g3_sym(a, b, c):
    return sum(
        _g3_unsym(*perm) for perm in
        ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a))
    ) / 6.0


# ---------------------------------------------------------------------------
# Linear-interpolation helper (matches scipy interp1d fill_value="extrapolate")
# ---------------------------------------------------------------------------
def _interp_extrap(x, xp, fp):
    """Linear interpolation + linear extrapolation at both ends."""
    result = jnp.interp(x, xp, fp)
    sl = (fp[1]  - fp[0])  / (xp[1]  - xp[0])
    sr = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    result = jnp.where(x < xp[0],  fp[0]  + sl * (x - xp[0]),  result)
    result = jnp.where(x > xp[-1], fp[-1] + sr * (x - xp[-1]), result)
    return result


# ---------------------------------------------------------------------------
# Core loop integral (JIT-compiled; dispatches to GPU)
# ---------------------------------------------------------------------------
@jit
def _loop_terms_jax(
    k_flat, mu_flat, p: _IvanovParamsJAX,
    q_arr, qx, qy, qz, q2, pq, weight,
    f_growth, k_lin, p_lin, eps,
):
    """
    JAX port of IvanovFullModel._loop_terms.

    k_flat, mu_flat : (n_ext,)   external k, mu values
    q_arr … weight  : (n_quad,)  precomputed quadrature nodes
    Returns p22, p13, each of shape (n_ext,).
    """
    kx = k_flat * jnp.sqrt(jnp.maximum(1.0 - mu_flat ** 2, 0.0))
    kz = k_flat * mu_flat

    dot_kq = kx[:, None] * qx[None, :] + kz[:, None] * qz[None, :]
    p2     = jnp.maximum(k_flat[:, None] ** 2 + q2[None, :] - 2.0 * dot_kq, eps)
    p_val  = jnp.sqrt(p2)
    pz     = kz[:, None] - qz[None, :]

    pp  = _interp_extrap(p_val.ravel(), k_lin, p_lin).reshape(p_val.shape)
    mu1 = _safe_div(qz[None, :],  q_arr[None, :], eps)
    mu2 = _safe_div(pz, p_val, eps)
    cos12         = _safe_div(dot_kq - q2[None, :], p_val * q_arr[None, :], eps)
    khat_dot_qhat = _safe_div(dot_kq, k_flat[:, None] * q_arr[None, :], eps)

    qvec = (
        jnp.broadcast_to(qx[None, :], p_val.shape),
        jnp.broadcast_to(qy[None, :], p_val.shape),
        jnp.broadcast_to(qz[None, :], p_val.shape),
    )
    pvec = (kx[:, None] - qx[None, :], -jnp.broadcast_to(qy[None, :], p_val.shape), pz)
    kvec = (kx[:, None], jnp.zeros_like(p_val), kz[:, None])
    mq   = (-qvec[0], -qvec[1], -qvec[2])

    f2 = _f2_sym(qvec, pvec)
    g2 = _g2_sym(qvec, pvec)
    f3 = _f3_sym(kvec, qvec, mq)
    g3 = _g3_sym(qvec, mq, kvec)

    k2q  = _safe_div(p_val, q_arr[None, :], eps) + _safe_div(q_arr[None, :], p_val, eps)
    emix = mu1 * mu2 * (
        mu2 ** 2 * _safe_div(p_val, q_arr[None, :], eps)
        + mu1 ** 2 * _safe_div(q_arr[None, :], p_val, eps)
    )

    K2 = (
        0.5 * p.b_delta2
        + p.b_G2 * (cos12 ** 2 - 1.0)
        + p.b1 * f2
        - p.b_eta * f_growth * mu_flat[:, None] ** 2 * g2
        - 0.5 * f_growth * p.b_delta_eta * (mu1 ** 2 + mu2 ** 2)
        + p.b_eta2 * f_growth ** 2 * mu1 ** 2 * mu2 ** 2
        + 0.5 * p.b1 * f_growth * mu1 * mu2 * k2q
        - 0.5 * p.b_eta * f_growth ** 2 * emix
        + p.b_KK_par * (mu1 * mu2 * cos12 - (mu1 ** 2 + mu2 ** 2) / 3.0 + 1.0 / 9.0)
        + p.b_Pi2_par * (mu1 * mu2 * cos12 + (5.0 / 7.0) * mu_flat[:, None] ** 2 * (1.0 - cos12 ** 2))
    )

    p22 = 2.0 * jnp.sum(K2 ** 2 * pq[None, :] * pp * weight[None, :], axis=1)

    qp2q2  = _safe_div(qz[None, :] ** 2, q2[None, :], eps)
    pp2p2  = _safe_div(pz ** 2, p2, eps)
    sel    = 1.0 - khat_dot_qhat ** 2
    dpq    = dot_kq - q2[None, :]
    pq_mix = _safe_div(dpq * pz * qz[None, :],      p2 * q2[None, :], eps)
    pq_shr = _safe_div(dpq * pz ** 2,               p2 * q2[None, :], eps)
    pq_crs = _safe_div(dpq * qz[None, :] * pz,      q2[None, :] * p2, eps)
    qp_par      = qz[None, :] * pz
    qp_q2       = _safe_div(qp_par, q2[None, :], eps)
    qp_p2       = _safe_div(qp_par, p2, eps)
    qp_q2p2     = _safe_div(qp_par, q2[None, :] * p2, eps)

    extra = (
        (4.0 / 21.0) * (5.0 * p.b_G2 + 2.0 * p.b_gamma3) * (cos12 ** 2 - 1.0)
        - (2.0 / 21.0) * f_growth * p.b_delta_eta * (3.0 * pp2p2 + 5.0 * qp2q2)
        + (4.0 / 7.0) * f_growth ** 2 * p.b_eta2 * qp2q2 * pp2p2
        + (20.0 / 21.0) * p.b_KK_par * (pq_mix - pp2p2 / 3.0 - qp2q2 / 3.0 + 1.0 / 9.0)
        + (10.0 / 21.0) * p.b_Pi2_par * pq_shr
        + (10.0 / 21.0) * (
            p.b_delta_Pi2_par - p.b_KPi2_par / 3.0 - f_growth * p.b_eta_Pi2_par * qp2q2
        ) * pp2p2
        + (10.0 / 21.0) * p.b_KPi2_par * pq_crs
        + (10.0 / 21.0) * f_growth * p.b_Pi2_par * _safe_div(
            qz[None, :] * pz ** 3, q2[None, :] * p2, eps
        )
        + (p.b_Pi3_par + 2.0 * p.b_Pi2_par) * (
            (13.0 / 21.0) * pq_crs
            - (5.0 / 9.0) * mu_flat[:, None] ** 2 * (cos12 ** 2 - 1.0 / 3.0)
        )
        + (2.0 / 21.0) * f_growth * p.b1 * (5.0 * qp_q2 + 3.0 * qp_p2)
        - (2.0 / 7.0) * f_growth ** 2 * p.b_eta * qp_q2p2 * (pz ** 2 + qz[None, :] ** 2)
    )

    i3 = (
        p.b1 * jnp.sum(f3 * pq[None, :] * weight[None, :], axis=1)
        - f_growth * p.b_eta * mu_flat ** 2 * jnp.sum(g3 * pq[None, :] * weight[None, :], axis=1)
        + jnp.sum(sel * pq[None, :] * extra * weight[None, :], axis=1)
    )

    pk = _interp_extrap(k_flat, k_lin, p_lin)
    k1 = p.b1 - p.b_eta * f_growth * mu_flat ** 2
    return p22, 6.0 * k1 * pk * i3


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------
class IvanovFullModelJAX:
    """
    JAX/GPU port of IvanovFullModel.

    Quadrature grid is built once in __init__ (on CPU); all loop-integral
    evaluations run on GPU via JIT-compiled _loop_terms_jax.

    API mirrors IvanovFullModel.evaluate_components so the two can be swapped.
    """

    def __init__(
        self,
        k_lin: np.ndarray,
        p_lin: np.ndarray,
        f_growth: float,
        *,
        qmin: float = 1.0e-3,
        qmax: float = 8.0,
        nq: int = 40,
        nmuq: int = 18,
        nphi: int = 12,
    ) -> None:
        self.f_growth = float(f_growth)
        self._eps     = 1.0e-12
        self.k_lin    = jnp.array(k_lin, dtype=jnp.float64)
        self.p_lin    = jnp.array(p_lin, dtype=jnp.float64)

        # Build Gauss-Legendre quadrature on log-q × mu × phi
        xq, wq = np.polynomial.legendre.leggauss(int(nq))
        xq = 0.5 * (xq + 1.0) * (np.log(qmax) - np.log(qmin)) + np.log(qmin)
        wq = 0.5 * (np.log(qmax) - np.log(qmin)) * wq
        q  = np.exp(xq)

        muq, wmu = np.polynomial.legendre.leggauss(int(nmuq))
        dphi = 2.0 * np.pi / max(int(nphi), 1)
        phi  = (np.arange(int(nphi)) + 0.5) * dphi
        wphi = np.full_like(phi, dphi)

        qq, mm, pp = np.meshgrid(q, muq, phi, indexing="ij")
        wqq, wmm, wpp = np.meshgrid(wq, wmu, wphi, indexing="ij")
        qp = np.sqrt(np.maximum(1.0 - mm ** 2, 0.0))

        self.q_arr  = jnp.array(qq.ravel())
        self.qx     = jnp.array((qq * qp * np.cos(pp)).ravel())
        self.qy     = jnp.array((qq * qp * np.sin(pp)).ravel())
        self.qz     = jnp.array((qq * mm).ravel())
        self.q2     = jnp.array((qq ** 2).ravel())
        self.pq     = jnp.array(np.interp(qq.ravel(), k_lin, p_lin))
        self.weight = jnp.array((qq ** 3 * wqq * wmm * wpp / (2.0 * np.pi) ** 3).ravel())

    def evaluate_components(
        self,
        k: np.ndarray,
        mu: np.ndarray,
        params: IvanovFullParams,
    ) -> dict[str, jnp.ndarray]:
        """
        Evaluate tree + loop + counterterm P3D on arbitrary (k, mu) arrays.

        Returns a dict with keys: 'tree', 'loop_22', 'loop_13', 'counterterm', 'total'.
        Each value has the same shape as the input k/mu arrays.
        Output arrays live on the JAX device; call np.asarray() to pull to CPU.
        """
        k_f   = jnp.asarray(k,  dtype=jnp.float64).ravel()
        mu_f  = jnp.asarray(mu, dtype=jnp.float64).ravel()
        shape = jnp.asarray(k, dtype=jnp.float64).shape
        jp    = _to_jax_params(params)

        pk   = _interp_extrap(k_f, self.k_lin, self.p_lin)
        k1   = jp.b1 - jp.b_eta * self.f_growth * mu_f ** 2
        tree = k1 ** 2 * pk

        p22, p13 = _loop_terms_jax(
            k_f, mu_f, jp,
            self.q_arr, self.qx, self.qy, self.qz,
            self.q2, self.pq, self.weight,
            self.f_growth, self.k_lin, self.p_lin, self._eps,
        )
        ct    = (jp.c0_ct + jp.c2_ct * mu_f ** 2 + jp.c4_ct * mu_f ** 4) * k_f ** 2 * pk
        total = tree + p22 + p13 + ct

        return {k: v.reshape(shape) for k, v in zip(
            ("tree", "loop_22", "loop_13", "counterterm", "total"),
            (tree, p22, p13, ct, total),
        )}
