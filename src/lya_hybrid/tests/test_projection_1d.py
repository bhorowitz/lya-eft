import numpy as np

from lya_hybrid.projection_1d import project_to_1d


def test_projection_matches_analytic_for_inverse_k2() -> None:
    # For P3D = A / k^2, P1D = A/(2pi) * ln(kmax / k_parallel).
    A = 2.75
    kmax = 10.0
    kpar = np.array([0.05, 0.1, 0.2, 0.5, 1.0])

    def p3d(k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        _ = mu
        return A / (k**2)

    projected = project_to_1d(kpar_values=kpar, p3d_callable=p3d, kmax_proj=kmax, nint=5000, method="trapz")
    expected = (A / (2.0 * np.pi)) * np.log(kmax / kpar)

    assert np.allclose(projected["raw"], expected, rtol=3e-3, atol=0.0)
