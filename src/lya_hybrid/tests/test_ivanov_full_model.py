from __future__ import annotations

import numpy as np

from lya_hybrid.model_ivanov_full import IvanovFullModel, IvanovFullParams


def test_full_model_returns_finite_components() -> None:
    k_lin = np.logspace(-3, 1, 256)
    p_lin = k_lin ** -2.2
    model = IvanovFullModel(k_lin, p_lin, f_growth=0.95, qmax=4.0, nq=8, nmuq=6, nphi=6)
    params = IvanovFullParams(
        b1=-0.4,
        b_eta=-0.3,
        b_delta2=-0.5,
        b_G2=-0.2,
        b_KK_par=0.4,
        b_delta_eta=-0.1,
        b_eta2=0.2,
        b_Pi2_par=0.3,
        b_Pi3_par=-0.4,
        b_delta_Pi2_par=0.5,
        b_eta_Pi2_par=0.2,
        b_KPi2_par=-0.3,
    )
    k_eval = np.array([0.08, 0.15, 0.30], dtype=float)
    mu_eval = np.array([0.0, 0.5, 0.9], dtype=float)
    out = model.evaluate_components(k_eval, mu_eval, params)
    for name in ("tree", "loop_22", "loop_13", "total"):
        assert np.all(np.isfinite(out[name]))
        assert out[name].shape == k_eval.shape
