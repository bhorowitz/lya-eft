import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
import jax.numpy as jnp
from scripts_jax.run_2405_stage1_sdss_baseline_jax import jax_dot, jax_norm, jax_add, jax_mul, model, project_to_1d, residual_fun
import numpy as np
import time

# --- Unit tests ---
def test_jax_dot():
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])
    assert jnp.allclose(jax_dot(a, b), 32.0)

def test_jax_norm():
    a = jnp.array([3.0, 4.0])
    assert jnp.allclose(jax_norm(a), 5.0)

def test_jax_add():
    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0, 4.0])
    assert jnp.allclose(jax_add(a, b), jnp.array([4.0, 6.0]))

def test_jax_mul():
    a = jnp.array([2.0, 3.0])
    b = jnp.array([4.0, 5.0])
    assert jnp.allclose(jax_mul(a, b), jnp.array([8.0, 15.0]))

def test_model():
    params = jnp.array([2.0, 1.0])
    k = jnp.array([1.0, 2.0])
    mu = jnp.array([0.5, -0.5])
    out = model(params, k, mu)
    expected = params[0] * k + params[1] * mu
    assert jnp.allclose(out, expected)

def test_project_to_1d():
    params = jnp.array([1.0, 0.5])
    k = jnp.array([1.0, 2.0])
    mu = jnp.array([0.0, 1.0])
    out = project_to_1d(params, k, mu)
    assert out.shape == (2,)

def test_residual_fun():
    params = jnp.array([1.0, 0.5])
    k = jnp.array([1.0, 2.0])
    mu = jnp.array([0.0, 1.0])
    data = jnp.array([1.5, 2.5])
    res = residual_fun(params, k, mu, data)
    assert res.shape == (2,)

# --- Benchmark speed ---
def benchmark_jax():
    params = jnp.array([1.0, 0.5])
    k = jnp.linspace(0.03, 3.0, 1000)
    mu = jnp.linspace(-1, 1, 100)
    data = jnp.ones_like(k)
    start = time.time()
    for _ in range(10):
        _ = project_to_1d(params, k, mu)
    jax_time = time.time() - start
    print(f"JAX project_to_1d (GPU=2): {jax_time:.4f} s for 10 runs")

# --- Compare accuracy with numpy ---
def benchmark_numpy():
    params = np.array([1.0, 0.5])
    k = np.linspace(0.03, 3.0, 1000)
    mu = np.linspace(-1, 1, 100)
    def np_model(params, k, mu):
        return params[0] * k + params[1] * mu
    def np_project_to_1d(params, k, mu):
        return np.sum(np_model(params, k[:, None], mu[None, :]), axis=-1)
    start = time.time()
    for _ in range(10):
        _ = np_project_to_1d(params, k, mu)
    np_time = time.time() - start
    print(f"Numpy project_to_1d (CPU): {np_time:.4f} s for 10 runs")

    # Compare accuracy
    jax_out = project_to_1d(jnp.array([1.0, 0.5]), jnp.array(k), jnp.array(mu))
    np_out = np_project_to_1d(params, k, mu)
    assert np.allclose(np_out, np.array(jax_out), atol=1e-6)

# Example loop integral: integrate f(q) = q^2 * exp(-q) over q=[0,10]
@jax.jit
def loop_integral_demo():
    q = jnp.linspace(0, 10, 1000)
    f_q = q**2 * jnp.exp(-q)
    dq = q[1] - q[0]
    return jnp.sum(f_q) * dq

def test_loop_integral():
    start = time.time()
    result = loop_integral_demo()
    elapsed = time.time() - start
    expected = 2.0  # Analytical: int_0^inf q^2 exp(-q) dq = 2
    print(f"Loop integral result: {result:.6f}, expected: {expected}, time: {elapsed:.4f} s")
    assert jnp.allclose(result, expected, atol=1e-2)

if __name__ == '__main__':
    test_jax_dot()
    test_jax_norm()
    test_jax_add()
    test_jax_mul()
    test_model()
    test_project_to_1d()
    test_residual_fun()
    benchmark_jax()
    benchmark_numpy()
    test_loop_integral()
    print("All tests and benchmarks passed.")
