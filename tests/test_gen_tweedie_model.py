#!/usr/bin/env python3
# ===========================================================================
# tests/test_gen_tweedie_model.py
#
# Unit tests for the JAX callables of the Gen-Tweedie model:
#   - pedestal PDF normalisation
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
    _pdf_extra,
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
#     Pedestal PDF
# ==============================


def test_pedestal_normalized():
    """Gaussian PDF integrates to 1 over a fine grid covering ~10 sigma."""
    ped_mean, ped_sigma = -600.0, 600.0
    xsp = np.linspace(-8000.0, 8000.0, 20001)
    dx = xsp[1] - xsp[0]
    pdf = np.asarray(_pdf_extra(jnp.asarray(xsp), jnp.array([ped_mean, ped_sigma])))
    assert np.isclose(pdf.sum() * dx, 1.0, rtol=1e-6)


def test_pedestal_matches_scipy():
    ped_mean, ped_sigma = -600.0, 600.0
    xsp = np.linspace(-5000.0, 5000.0, 1001)
    pdf_jax = np.asarray(_pdf_extra(jnp.asarray(xsp), jnp.array([ped_mean, ped_sigma])))
    pdf_ref = norm.pdf(xsp, loc=ped_mean, scale=ped_sigma)
    assert np.max(np.abs(pdf_jax - pdf_ref)) < 1e-14


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
    """Inverse-FFT of the analytic Gamma CF must reproduce the Gamma PDF.

    We use a symmetric grid with q=0 at index 0 (standard FFT layout).
    """
    spe_mean, spe_sigma = 6000.0, 800.0
    alpha_true = (spe_mean / spe_sigma) ** 2
    theta_true = spe_mean / alpha_true

    N = 2048
    dq = 20.0
    q = np.arange(N) * dq
    # frequencies corresponding to q
    freq = 2.0 * np.pi * np.fft.fftfreq(N, d=dq)

    a, b = reparam_from_spe(spe_mean, spe_sigma)
    spe = jnp.array([a, b, 0.1])
    cf = np.asarray(_ser_ft(jnp.asarray(freq), spe))

    # inverse-FFT -> PDF on q grid
    pdf_from_cf = np.real(np.fft.ifft(cf)) / dq

    pdf_ref = gamma.pdf(q, a=alpha_true, scale=theta_true)
    # Only compare where both are appreciable (Gamma support = q > 0)
    mask = q > 500.0
    max_err = np.max(np.abs(pdf_from_cf[mask] - pdf_ref[mask]))
    assert max_err < 1e-4, f"max err {max_err}"


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
                jnp.array([0.0, 0.0, xi]),  # xi is slot 2
            )
            assert np.isclose(
                float(val.real), 1.0, atol=1e-10
            ), f"xi={xi} lam={lam}: pgf(1)={val}"
            assert abs(float(val.imag)) < 1e-10


def test_gen_poisson_pgf_reduces_to_poisson():
    """As xi -> 0, Gen-Poisson pgf -> exp(lam * (s - 1)) (plain Poisson)."""
    lam = 1.5
    s = 0.7 + 0.3j
    # very small xi
    val_gen = _count_pgf(jnp.asarray(s), jnp.asarray(lam), jnp.array([0.0, 0.0, 1e-8]))
    val_ref = np.exp(lam * (s - 1.0))
    assert np.isclose(complex(val_gen), val_ref, rtol=1e-6, atol=1e-6)


def test_gen_poisson_pgf_derivatives_at_zero():
    """The pgf derivatives at s=0 should give the PMF:  pgf^(n)(0)/n! = p_n."""
    lam = 1.2
    xi = 0.1
    # compute p_n from the explicit PMF
    p_0 = float(_count_pmf(jnp.asarray(lam), jnp.asarray(xi), 0))
    # pgf(0) = exp(-lam) since T(0) = -W(0)/xi = 0
    val = _count_pgf(jnp.asarray(0.0 + 0j), jnp.asarray(lam), jnp.array([0.0, 0.0, xi]))
    assert np.isclose(float(val.real), p_0, atol=1e-12)


# ==============================
#     Gen-Poisson PMF sums to 1
# ==============================


def test_count_pmf_sums_to_one():
    for lam, xi in [(0.5, 0.1), (1.5, 0.1), (3.0, 0.05), (2.0, 0.3)]:
        total = 0.0
        for n in range(100):
            total += float(_count_pmf(jnp.asarray(lam), jnp.asarray(xi), n))
        assert np.isclose(total, 1.0, atol=1e-8), f"lam={lam}, xi={xi}: sum={total}"
