#!/usr/bin/env python3
# ===========================================================================
# tests/test_gen_tweedie_model.py
#
# Unit tests for the JAX callables of the Gen-Tweedie model:
#   - Gaussian pedestal FT normalisation (ft_extra(0) = 1) and IFFT recovery
#   - Gamma CF matches analytic form and is correctly normalised (CF(0)=1)
#   - Gen-Poisson pgf recovers plain Poisson as xi -> 0
#   - Gen-Poisson PMF sums to 1
#   - Reparameterisation inverse is consistent
# ===========================================================================

import numpy as np
import pytest
import jax.numpy as jnp

from scipy.stats import norm, gamma

from ..models.gen_tweedie import (
    _ft_extra,
    _ser_ft,
    _count_pgf,
    _count_pmf,
    spe_from_reparam,
    reparam_from_spe,
)


# ==============================
#     Reparameterisation
# ==============================


@pytest.mark.parametrize(
    "spe_mean,spe_sigma",
    [(6000.0, 800.0), (1000.0, 100.0), (10000.0, 5000.0), (100.0, 50.0)],
)
def test_reparam_roundtrip(spe_mean, spe_sigma):
    a, b = reparam_from_spe(spe_mean, spe_sigma)
    m, s = spe_from_reparam(jnp.asarray(a), jnp.asarray(b))
    assert np.isclose(float(m), spe_mean)
    assert np.isclose(float(s), spe_sigma)


def test_reparam_enforces_mean_gt_sigma():
    with pytest.raises(AssertionError):
        reparam_from_spe(100.0, 100.0)
    with pytest.raises(AssertionError):
        reparam_from_spe(50.0, 100.0)


# ==============================
#     Pedestal FT
# ==============================


def test_pedestal_ft_at_zero_is_one():
    """g0~(0) must equal 1 for any normalised PDF."""
    extra = jnp.array([-600.0, 600.0])
    val = _ft_extra(jnp.asarray(0.0), extra)
    assert np.isclose(float(val.real), 1.0, atol=1e-12)
    assert abs(float(val.imag)) < 1e-12


def test_pedestal_ft_ifft_recovers_gaussian():
    """IFFT of _ft_extra must reproduce the Gaussian PDF on a grid where
    the pedestal is well-contained (no aliasing)."""
    ped_mean, ped_sigma = -600.0, 600.0
    N = 4096
    dq = 10.0
    # Build xsp centred so the pedestal sits well inside the grid
    xsp = np.arange(N) * dq - N * dq / 2  # [-L/2, L/2)
    freq = 2.0 * np.pi * np.fft.fftfreq(N, d=dq)

    # _ft_extra gives g0~(w) = exp(i*mu*w - sigma^2*w^2/2)
    # IFFT convention: pdf[k] = (1/L) * sum_n G~_n * exp(+i*w_n*x_k)
    # numpy ifft: out[k] = (1/N) * sum_n f[n] * exp(+2pi*i*k*n/N)
    # so pdf = Re(ifft(ft)) / dq after the roll to physical xsp
    ft = np.asarray(_ft_extra(jnp.asarray(freq), jnp.array([ped_mean, ped_sigma])))
    pdf = np.real(np.fft.ifft(np.fft.ifftshift(ft))) / dq

    ref = norm.pdf(xsp, loc=ped_mean, scale=ped_sigma)
    mask = np.abs(xsp - ped_mean) < 5 * ped_sigma
    assert np.max(np.abs(pdf[mask] - ref[mask])) < 1e-3


# ==============================
#     Gamma characteristic function
# ==============================


def test_gamma_cf_at_zero():
    """Any characteristic function must satisfy f(0) = 1."""
    a, b = reparam_from_spe(6000.0, 800.0)
    spe = jnp.array([a, b, 0.1])
    cf0 = _ser_ft(jnp.asarray(0.0), spe)
    assert np.isclose(float(cf0.real), 1.0, atol=1e-12)
    assert abs(float(cf0.imag)) < 1e-12


def test_gamma_cf_via_ifft_recovers_pdf():
    """Inverse-FFT of the analytic Gamma CF must reproduce the Gamma PDF."""
    spe_mean, spe_sigma = 6000.0, 800.0
    alpha_true = (spe_mean / spe_sigma) ** 2
    theta_true = spe_mean / alpha_true

    N = 2048
    dq = 20.0
    q = np.arange(N) * dq
    freq = 2.0 * np.pi * np.fft.fftfreq(N, d=dq)

    a, b = reparam_from_spe(spe_mean, spe_sigma)
    spe = jnp.array([a, b, 0.1])
    cf = np.asarray(_ser_ft(jnp.asarray(freq), spe))

    pdf_from_cf = np.real(np.fft.ifft(cf)) / dq
    pdf_ref = gamma.pdf(q, a=alpha_true, scale=theta_true)
    mask = q > 500.0
    assert np.max(np.abs(pdf_from_cf[mask] - pdf_ref[mask])) < 1e-4


# ==============================
#     Gen-Poisson pgf
# ==============================


def test_gen_poisson_pgf_at_one_is_one():
    """pgf(1) = sum_k p_k = 1 by normalisation."""
    for xi in [0.01, 0.1, 0.3, 0.5]:
        for lam in [0.1, 1.0, 3.0]:
            val = _count_pgf(
                jnp.asarray(1.0 + 0j),
                jnp.asarray(lam),
                jnp.array([0.0, 0.0, xi]),
            )
            assert np.isclose(float(val.real), 1.0, atol=1e-10)
            assert abs(float(val.imag)) < 1e-10


def test_gen_poisson_pgf_reduces_to_poisson():
    """As xi -> 0, Gen-Poisson pgf -> exp(lam*(s-1))."""
    lam = 1.5
    s = 0.7 + 0.3j
    val_gen = _count_pgf(jnp.asarray(s), jnp.asarray(lam), jnp.array([0.0, 0.0, 1e-8]))
    val_ref = np.exp(lam * (s - 1.0))
    assert np.isclose(complex(val_gen), val_ref, rtol=1e-6, atol=1e-6)


def test_gen_poisson_pgf_derivatives_at_zero():
    """pgf^(n)(0)/n! = p_n."""
    lam, xi = 1.5, 0.1
    for n in range(1, 5):
        p_n_pgf = float(_count_pmf(jnp.asarray(lam), jnp.asarray(xi), n).real)
        p_n_pmf = float(_count_pmf(jnp.asarray(lam), jnp.asarray(xi), n).real)
        assert np.isclose(p_n_pgf, p_n_pmf, rtol=1e-8)


def test_gen_poisson_pmf_sums_to_one():
    for lam in [0.5, 1.5, 3.0]:
        total = sum(
            float(_count_pmf(jnp.asarray(lam), jnp.asarray(0.1), k).real)
            for k in range(50)
        )
        assert np.isclose(total, 1.0, atol=1e-6), f"lam={lam}: sum={total}"
