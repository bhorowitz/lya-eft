import numpy as np

from lya_hybrid.model_hybrid import HybridToyModel, HybridToyParams
from lya_hybrid.model_ivanov import IvanovToyModel, IvanovToyParams


def test_hybrid_reduces_to_ivanov_when_sigma_and_bt_zero() -> None:
    k = np.logspace(-2, 0.4, 50)
    mu = np.linspace(0.0, 1.0, 40)
    kk = np.broadcast_to(k[None, :], (mu.size, k.size))
    mm = np.broadcast_to(mu[:, None], (mu.size, k.size))

    p_lin = 1.0 / (1.0 + k**1.5)
    f_growth = 0.97

    iv = IvanovToyModel(k, p_lin, f_growth)
    hy = HybridToyModel(k, p_lin, f_growth, k_t=0.5)

    iv_params = IvanovToyParams(
        b1=-0.24,
        b_eta=-0.33,
        c0=0.01,
        c2=0.02,
        c4=0.015,
        loop_amp=0.08,
        loop_mu2=0.6,
        loop_mu4=0.2,
        loop_k_nl=1.7,
    )

    hy_params = HybridToyParams(
        b_delta=iv_params.b1,
        b_eta=iv_params.b_eta,
        b_t=0.0,
        c0=iv_params.c0,
        c2=iv_params.c2,
        c4=iv_params.c4,
        loop_amp=iv_params.loop_amp,
        loop_mu2=iv_params.loop_mu2,
        loop_mu4=iv_params.loop_mu4,
        loop_k_nl=iv_params.loop_k_nl,
        sigma_th=0.0,
    )

    p_iv = iv.evaluate_components(kk, mm, iv_params)["total"]
    p_hy = hy.evaluate_components(kk, mm, hy_params)["total"]

    assert np.allclose(p_iv, p_hy, rtol=1e-10, atol=1e-12)
