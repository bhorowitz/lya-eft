from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d


def _vec(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return x, y, z


def _add(a: tuple[np.ndarray, np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _dot(a: tuple[np.ndarray, np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm2(a: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return _dot(a, a)


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1.0e-20) -> np.ndarray:
    num_b, den_b = np.broadcast_arrays(num, den)
    out = np.zeros_like(num_b, dtype=float)
    np.divide(num_b, den_b, out=out, where=np.abs(den_b) > eps)
    return out


def _alpha(a: tuple[np.ndarray, np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return _safe_div(_dot(_add(a, b), a), _norm2(a))


def _beta(a: tuple[np.ndarray, np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    ab = _add(a, b)
    return _safe_div(_norm2(ab) * _dot(a, b), 2.0 * _norm2(a) * _norm2(b))


def _f2_unsym(a: tuple[np.ndarray, np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return (5.0 / 7.0) * _alpha(a, b) + (2.0 / 7.0) * _beta(a, b)


def _g2_unsym(a: tuple[np.ndarray, np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return (3.0 / 7.0) * _alpha(a, b) + (4.0 / 7.0) * _beta(a, b)


def _f2_sym(a: tuple[np.ndarray, np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return 0.5 * (_f2_unsym(a, b) + _f2_unsym(b, a))


def _g2_sym(a: tuple[np.ndarray, np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return 0.5 * (_g2_unsym(a, b) + _g2_unsym(b, a))


def _f3_unsym(
    a: tuple[np.ndarray, np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray, np.ndarray],
    c: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    bc = _add(b, c)
    ab = _add(a, b)
    term1 = 7.0 * _alpha(a, bc) * _f2_unsym(b, c) + 2.0 * _beta(a, bc) * _g2_unsym(b, c)
    term2 = _g2_unsym(a, b) * (7.0 * _alpha(ab, c) + 2.0 * _beta(ab, c))
    return (term1 + term2) / 18.0


def _g3_unsym(
    a: tuple[np.ndarray, np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray, np.ndarray],
    c: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    bc = _add(b, c)
    ab = _add(a, b)
    term1 = 3.0 * _alpha(a, bc) * _f2_unsym(b, c) + 6.0 * _beta(a, bc) * _g2_unsym(b, c)
    term2 = _g2_unsym(a, b) * (3.0 * _alpha(ab, c) + 6.0 * _beta(ab, c))
    return (term1 + term2) / 18.0


def _f3_sym(
    a: tuple[np.ndarray, np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray, np.ndarray],
    c: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    perms = (
        (a, b, c),
        (a, c, b),
        (b, a, c),
        (b, c, a),
        (c, a, b),
        (c, b, a),
    )
    return sum(_f3_unsym(*perm) for perm in perms) / 6.0


def _g3_sym(
    a: tuple[np.ndarray, np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray, np.ndarray],
    c: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    perms = (
        (a, b, c),
        (a, c, b),
        (b, a, c),
        (b, c, a),
        (c, a, b),
        (c, b, a),
    )
    return sum(_g3_unsym(*perm) for perm in perms) / 6.0


@dataclass
class IvanovFullParams:
    b1: float
    b_eta: float
    b_delta2: float
    b_G2: float
    b_KK_par: float
    b_delta_eta: float
    b_eta2: float
    b_Pi2_par: float
    b_gamma3: float = 0.0
    b_delta_Pi2_par: float = 0.0
    b_eta_Pi2_par: float = 0.0
    b_KPi2_par: float = 0.0
    b_Pi3_par: float = 0.0


class IvanovFullModel:
    """Direct numerical implementation of the supplement kernels for the full one-loop LyA basis."""

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
        self._interp = interp1d(k_lin, p_lin, kind="linear", fill_value="extrapolate")
        self._eps = 1.0e-12

        xq, wq = np.polynomial.legendre.leggauss(int(nq))
        xq = 0.5 * (xq + 1.0) * (np.log(qmax) - np.log(qmin)) + np.log(qmin)
        wq = 0.5 * (np.log(qmax) - np.log(qmin)) * wq
        q = np.exp(xq)

        muq, wmu = np.polynomial.legendre.leggauss(int(nmuq))
        # Use azimuthal midpoints instead of sampling exactly at phi=0. In the
        # direct tensor-product loop quadrature, phi=0 can line up the internal
        # and external wavevectors and produce spurious spikes from the
        # p=|k-q|->0 corner, even though that set has measure zero.
        dphi = 2.0 * np.pi / max(int(nphi), 1)
        phi = (np.arange(int(nphi), dtype=float) + 0.5) * dphi
        wphi = np.full_like(phi, dphi, dtype=float)

        qq, mm, pp = np.meshgrid(q, muq, phi, indexing="ij")
        wqq, wmm, wpp = np.meshgrid(wq, wmu, wphi, indexing="ij")

        q_perp = np.sqrt(np.maximum(1.0 - mm**2, 0.0))
        self.q = qq.ravel()
        self.qx = (qq * q_perp * np.cos(pp)).ravel()
        self.qy = (qq * q_perp * np.sin(pp)).ravel()
        self.qz = (qq * mm).ravel()
        self.qmu = mm.ravel()
        self.q2 = self.q**2
        self.pq = np.asarray(self._interp(self.q), dtype=float)
        self.weight = (qq**3 * wqq * wmm * wpp / (2.0 * np.pi) ** 3).ravel()

    def _khat_dot_qhat(self, k: np.ndarray, dot_kq: np.ndarray) -> np.ndarray:
        return _safe_div(dot_kq, k[:, None] * self.q[None, :], eps=self._eps)

    def _loop_terms(self, k: np.ndarray, mu: np.ndarray, params: IvanovFullParams) -> tuple[np.ndarray, np.ndarray]:
        kx = k * np.sqrt(np.maximum(1.0 - mu**2, 0.0))
        kz = k * mu

        dot_kq = kx[:, None] * self.qx[None, :] + kz[:, None] * self.qz[None, :]
        p2 = np.maximum(k[:, None] ** 2 + self.q2[None, :] - 2.0 * dot_kq, self._eps)
        p = np.sqrt(p2)
        pz = kz[:, None] - self.qz[None, :]
        pp = np.asarray(self._interp(p.ravel()), dtype=float).reshape(p.shape)

        mu1 = _safe_div(self.qz[None, :], self.q[None, :], eps=self._eps)
        mu2 = _safe_div(pz, p, eps=self._eps)
        cos12 = _safe_div(dot_kq - self.q2[None, :], p * self.q[None, :], eps=self._eps)
        khat_dot_qhat = self._khat_dot_qhat(k, dot_kq)

        qvec = _vec(np.broadcast_to(self.qx[None, :], p.shape), np.broadcast_to(self.qy[None, :], p.shape), np.broadcast_to(self.qz[None, :], p.shape))
        pvec = _vec(kx[:, None] - self.qx[None, :], -self.qy[None, :], kz[:, None] - self.qz[None, :])
        kvec = _vec(kx[:, None], np.zeros_like(p), kz[:, None])
        minus_qvec = _vec(-qvec[0], -qvec[1], -qvec[2])

        f2 = _f2_sym(qvec, pvec)
        g2 = _g2_sym(qvec, pvec)
        f3 = _f3_sym(kvec, qvec, minus_qvec)
        g3 = _g3_sym(qvec, minus_qvec, kvec)

        k2_over_q = _safe_div(p, self.q[None, :], eps=self._eps) + _safe_div(self.q[None, :], p, eps=self._eps)
        eta_mix = mu1 * mu2 * (
            mu2 * _safe_div(p, self.q[None, :], eps=self._eps)
            + mu1 * _safe_div(self.q[None, :], p, eps=self._eps)
        )
        K2 = (
            0.5 * params.b_delta2
            + params.b_G2 * (cos12**2 - 1.0)
            + params.b1 * f2
            - params.b_eta * self.f_growth * mu[:, None] ** 2 * g2
            - 0.5 * self.f_growth * params.b_delta_eta * (mu1**2 + mu2**2)
            + params.b_eta2 * self.f_growth**2 * mu1**2 * mu2**2
            + 0.5 * params.b1 * self.f_growth * mu1 * mu2 * k2_over_q
            - 0.5 * params.b_eta * self.f_growth**2 * eta_mix
            + params.b_KK_par * (mu1 * mu2 * cos12 - (mu1**2 + mu2**2) / 3.0 + 1.0 / 9.0)
            + params.b_Pi2_par * (mu1 * mu2 * cos12 + (5.0 / 7.0) * mu[:, None] ** 2 * (1.0 - cos12**2))
        )

        p22 = 2.0 * np.sum((K2**2) * self.pq[None, :] * pp * self.weight[None, :], axis=1)

        qpar2_over_q2 = _safe_div(self.qz[None, :] ** 2, self.q2[None, :], eps=self._eps)
        ppar2_over_p2 = _safe_div(pz**2, p2, eps=self._eps)
        selector = 1.0 - khat_dot_qhat**2
        dot_pq = dot_kq - self.q2[None, :]
        pq_mixed = _safe_div(dot_pq * pz * self.qz[None, :], p2 * self.q2[None, :], eps=self._eps)
        pq_shear = _safe_div(dot_pq * pz**2, p2 * self.q2[None, :], eps=self._eps)
        pq_cross = _safe_div(dot_pq * self.qz[None, :] * pz, self.q2[None, :] * p2, eps=self._eps)
        qpar_ppar = self.qz[None, :] * pz
        qpar_ppar_over_q2 = _safe_div(qpar_ppar, self.q2[None, :], eps=self._eps)
        qpar_ppar_over_p2 = _safe_div(qpar_ppar, p2, eps=self._eps)
        qpar_ppar_over_q2p2 = _safe_div(qpar_ppar, self.q2[None, :] * p2, eps=self._eps)

        extra = (
            (4.0 / 21.0) * (5.0 * params.b_G2 + 2.0 * params.b_gamma3) * (cos12**2 - 1.0)
            - (2.0 / 21.0) * self.f_growth * params.b_delta_eta * (3.0 * ppar2_over_p2 + 5.0 * qpar2_over_q2)
            + (4.0 / 7.0) * self.f_growth**2 * params.b_eta2 * qpar2_over_q2 * ppar2_over_p2
            + (20.0 / 21.0)
            * params.b_KK_par
            * (pq_mixed - ppar2_over_p2 / 3.0 - qpar2_over_q2 / 3.0 + 1.0 / 9.0)
            + (10.0 / 21.0) * params.b_Pi2_par * pq_shear
            + (10.0 / 21.0)
            * (
                params.b_delta_Pi2_par
                - params.b_KPi2_par / 3.0
                - self.f_growth * params.b_eta_Pi2_par * qpar2_over_q2
            )
            * ppar2_over_p2
            + (10.0 / 21.0) * params.b_KPi2_par * pq_cross
            + (10.0 / 21.0)
            * self.f_growth
            * params.b_Pi2_par
            * _safe_div(self.qz[None, :] * pz**3, self.q2[None, :] * p2, eps=self._eps)
            + (params.b_Pi3_par + 2.0 * params.b_Pi2_par)
            * (
                (13.0 / 21.0) * pq_cross
                - (5.0 / 9.0) * mu[:, None] ** 2 * (cos12**2 - 1.0 / 3.0)
            )
            + (2.0 / 21.0)
            * self.f_growth
            * params.b1
            * (5.0 * qpar_ppar_over_q2 + 3.0 * qpar_ppar_over_p2)
            - (2.0 / 7.0)
            * self.f_growth**2
            * params.b_eta
            * qpar_ppar_over_q2p2
            * (pz**2 + self.qz[None, :] ** 2)
        )

        i3 = (
            params.b1 * np.sum(f3 * self.pq[None, :] * self.weight[None, :], axis=1)
            - self.f_growth
            * params.b_eta
            * mu**2
            * np.sum(g3 * self.pq[None, :] * self.weight[None, :], axis=1)
            + np.sum(selector * self.pq[None, :] * extra * self.weight[None, :], axis=1)
        )

        k1 = params.b1 - params.b_eta * self.f_growth * mu**2
        pk = np.asarray(self._interp(k), dtype=float)
        p13 = 6.0 * k1 * pk * i3
        return p22, p13

    def evaluate_components(
        self,
        k: np.ndarray,
        mu: np.ndarray,
        params: IvanovFullParams,
    ) -> dict[str, np.ndarray]:
        k_arr = np.asarray(k, dtype=float)
        mu_arr = np.asarray(mu, dtype=float)
        shape = k_arr.shape
        if mu_arr.shape != shape:
            raise ValueError("k and mu must have identical shapes.")

        k_flat = k_arr.ravel()
        mu_flat = mu_arr.ravel()
        pk = np.asarray(self._interp(k_flat), dtype=float)
        k1 = params.b1 - params.b_eta * self.f_growth * mu_flat**2
        tree = k1**2 * pk
        p22, p13 = self._loop_terms(k_flat, mu_flat, params)
        total = tree + p22 + p13
        return {
            "tree": tree.reshape(shape),
            "loop_22": p22.reshape(shape),
            "loop_13": p13.reshape(shape),
            "total": total.reshape(shape),
        }
