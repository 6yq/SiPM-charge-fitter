# ===========================================================================
# tests/test_lambert_w.py
#
# Correctness and autodiff tests for the JAX Lambert-W kernel.
# Reference: scipy.special.lambertw on the principal branch (k=0).
# ===========================================================================

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from scipy.special import lambertw

from ..core.lambert_w import lambert_w0


# ==============================
#     Real-axis correctness
# ==============================


@pytest.mark.parametrize(
    "z",
    [1e-12, 1e-8, 0.001, 0.01, 0.1, 0.3, 0.5, 0.9, 1.0, 1.5, 2.0, 5.0, 50.0, 1e3, 1e6],
)
def test_real_axis_matches_scipy(z):
    w_jax = complex(lambert_w0(jnp.asarray(z)))
    w_ref = complex(lambertw(z, k=0))
    assert np.isclose(
        w_jax, w_ref, rtol=1e-12, atol=1e-14
    ), f"z={z}: jax={w_jax}, scipy={w_ref}"


@pytest.mark.parametrize("z", [-0.3, -0.2, -0.1, -1e-3])
def test_real_axis_negative_matches_scipy(z):
    """Principal branch on (-1/e, 0) is real."""
    w_jax = complex(lambert_w0(jnp.asarray(z)))
    w_ref = complex(lambertw(z, k=0))
    assert np.isclose(w_jax, w_ref, rtol=1e-12, atol=1e-14)


# ==============================
#     Complex-plane correctness
# ==============================


def test_complex_random_samples():
    """Random complex inputs within the reliable regime of the current
    initial-guess scheme.  |z| <= 1 comfortably covers the Gen-Poisson
    pgf argument range (|z| < xi * exp(-xi) < 0.37) with margin."""
    rng = np.random.default_rng(0)
    z = (rng.normal(size=200) + 1j * rng.normal(size=200)) * 0.7
    # clip |z| to 1.0 just in case of a large outlier
    z = np.where(np.abs(z) > 1.0, z / np.abs(z), z)
    w_jax = np.asarray(lambert_w0(jnp.asarray(z)))
    w_ref = np.asarray(lambertw(z, k=0))
    max_err = np.max(np.abs(w_jax - w_ref))
    assert max_err < 1e-12, f"max complex error {max_err}"


def test_gen_poisson_physical_range():
    """The exact argument range encountered in the Gen-Poisson pgf."""
    rng = np.random.default_rng(1)
    for xi in [1e-3, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9]:
        # |s| <= 1 for any Gamma CF on the real frequency axis
        s = rng.normal(size=100) + 1j * rng.normal(size=100)
        s = s / np.maximum(np.abs(s), 1.0)  # |s| <= 1
        arg = -xi * s * np.exp(-xi)
        w_jax = np.asarray(lambert_w0(jnp.asarray(arg)))
        w_ref = np.asarray(lambertw(arg, k=0))
        assert np.max(np.abs(w_jax - w_ref)) < 1e-12


# ==============================
#     Autodiff correctness
# ==============================


@pytest.mark.parametrize("z0", [0.05, 0.1, 0.5, 1.0, 2.0, 10.0])
def test_real_gradient_matches_analytic(z0):
    """dW/dz = W / (z * (1 + W)) on the principal branch."""

    def f(z):
        return jnp.real(lambert_w0(z + 0j))

    grad_jax = float(jax.grad(f)(z0))
    w = complex(lambertw(z0, k=0))
    grad_ref = float((w / (z0 * (1.0 + w))).real)
    assert np.isclose(grad_jax, grad_ref, rtol=1e-10, atol=1e-14)


def test_jit_and_vmap():
    """Kernel must be jit-compatible and vmap-compatible."""
    zs = jnp.linspace(0.01, 5.0, 100) + 0j
    f_jit = jax.jit(lambert_w0)
    w1 = np.asarray(f_jit(zs))
    w2 = np.asarray(jax.vmap(lambert_w0)(zs))
    w_ref = np.asarray(lambertw(np.asarray(zs), k=0))
    assert np.max(np.abs(w1 - w_ref)) < 1e-12
    assert np.max(np.abs(w2 - w_ref)) < 1e-12


# ==============================
#     Defining identity
# ==============================


def test_wew_equals_z():
    """W(z) * exp(W(z)) = z by definition (the fundamental test)."""
    rng = np.random.default_rng(2)
    z = (rng.normal(size=50) + 1j * rng.normal(size=50)) * 1.5
    w = np.asarray(lambert_w0(jnp.asarray(z)))
    residual = w * np.exp(w) - z
    assert np.max(np.abs(residual)) < 1e-12
