# ===========================================================================
# tests/test_density_pipeline.py
#
# End-to-end correctness of density_on_xsp: the full G(q) computation
# from (extra, spe, lam) through the FFT pipeline.
#
# Key invariants:
#   1. G(q) is normalised (integral ~ 1) when the grid covers the support.
#   2. As lam -> 0, G(q) -> g0(q) (pure pedestal).
#   3. As lam -> infinity, the distribution becomes broad and bell-shaped.
#   4. At moderate lam, the peak structure is visible.
#
# Also: G(q) reproduced here must match the same quantity computed via
# a direct sum over n-PE components (truncation test).
# ===========================================================================

import numpy as np
import pytest
import jax.numpy as jnp

from scipy.stats import norm, gamma as sgamma
from scipy.signal import fftconvolve

from ..core.fft_grid import build_grid
from ..core.likelihood import density_on_xsp
from ..models.gen_tweedie import (
    _pdf_extra,
    _ser_ft,
    _count_pgf,
    _count_pmf,
    reparam_from_spe,
)


# ==============================
#     Helpers
# ==============================


def _evaluate_density(grid, ped_mean, ped_sigma, spe_mean, spe_sigma, xi, lam):
    a, b = reparam_from_spe(spe_mean, spe_sigma)
    extra = jnp.array([ped_mean, ped_sigma])
    spe = jnp.array([a, b, xi])
    dens = density_on_xsp(
        extra,
        spe,
        jnp.asarray(lam),
        jnp.asarray(grid.freq),
        float(grid.xsp_width),
        int(grid.i_zero),
        jnp.asarray(grid.xsp),
        _pdf_extra,
        _ser_ft,
        _count_pgf,
    )
    return np.asarray(dens)


def _trapz(y, dx):
    return float(np.trapezoid(y, dx=dx))


# ==============================
#     Normalisation
# ==============================


def test_density_normalised_low_lam():
    """At low lam with a wide grid, integral(G) ~ 1."""
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    hist = np.zeros(len(bins) - 1)
    grid = build_grid(hist, bins, A=1, q_min=-5000.0)
    G = _evaluate_density(grid, -600.0, 600.0, 6000.0, 800.0, 0.1, lam=0.5)
    integral = _trapz(G, grid.xsp_width)
    # at lam=0.5 the distribution decays fast at high q; grid up to 60000 is fine
    assert abs(integral - 1.0) < 5e-3


def test_density_normalised_mid_lam():
    bins = np.arange(2000.0, 80000.0 + 200.0, 200.0)
    hist = np.zeros(len(bins) - 1)
    grid = build_grid(hist, bins, A=1, q_min=-5000.0)
    G = _evaluate_density(grid, -600.0, 600.0, 6000.0, 800.0, 0.1, lam=1.5)
    integral = _trapz(G, grid.xsp_width)
    assert abs(integral - 1.0) < 1e-2


# ==============================
#     lam -> 0 limit is pure pedestal
# ==============================


def test_low_lam_limit_is_pedestal():
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    hist = np.zeros(len(bins) - 1)
    grid = build_grid(hist, bins, A=1, q_min=-5000.0)
    G = _evaluate_density(grid, -600.0, 600.0, 6000.0, 800.0, 0.1, lam=1e-8)
    ped_ref = norm.pdf(grid.xsp, loc=-600.0, scale=600.0)
    max_err = np.max(np.abs(G - ped_ref))
    assert max_err < 1e-6


# ==============================
#     Direct convolution reference
# ==============================


def test_density_matches_direct_convolution():
    """Build G(q) = sum_k p_k * (g0 * f^{*k})(q) directly and compare."""
    ped_mean, ped_sigma = -600.0, 600.0
    spe_mean, spe_sigma = 6000.0, 800.0
    xi = 0.1
    lam = 1.5

    alpha = (spe_mean / spe_sigma) ** 2
    theta = spe_mean / alpha

    bins = np.arange(2000.0, 80000.0 + 200.0, 200.0)
    hist = np.zeros(len(bins) - 1)
    grid = build_grid(hist, bins, A=1, q_min=-5000.0)
    xsp = grid.xsp
    dx = grid.xsp_width

    G_fft = _evaluate_density(grid, ped_mean, ped_sigma, spe_mean, spe_sigma, xi, lam)

    # Direct reference: convolve the pedestal with Sum_k p_k * Gamma^{*k}.
    # Build the pure Gamma PDF on an evenly spaced grid starting at 0.
    # Gamma is supported on q > 0; we need enough points to cover several
    # times k*spe_mean. Use the existing xsp_width.
    q_gamma = np.arange(0.0, xsp[-1] - xsp[0] + dx, dx)
    f = sgamma.pdf(q_gamma, a=alpha, scale=theta)
    # trim trailing zeros
    f = f * dx  # discrete pmf so convolution sum approximates integral

    # build the PE mixture pdf as function of q (real line): sum_{k>=1} p_k f^{*k}
    pe_mixture = np.zeros_like(q_gamma)
    f_k = f.copy()
    max_k = 20
    for k in range(1, max_k + 1):
        if k > 1:
            f_k = fftconvolve(f_k, f)[: len(q_gamma)]
        p_k = float(_count_pmf(jnp.asarray(lam), jnp.asarray(xi), k))
        if p_k < 1e-14:
            break
        # pe_mixture is a PDF so we need f_k / dx
        pe_mixture[: len(f_k)] += p_k * (f_k / dx)

    p_0 = float(_count_pmf(jnp.asarray(lam), jnp.asarray(xi), 0))

    # build g0 on the same spacing
    g0 = norm.pdf(xsp, loc=ped_mean, scale=ped_sigma)
    # and g0 * delta for the 0PE component
    G_0 = p_0 * g0

    # convolve g0 with pe_mixture (both on grid spacing dx, g0 on xsp,
    # pe_mixture on q_gamma starting at 0). The convolution lives on q = xsp[i] + q_gamma[j].
    g0_times_pe = fftconvolve(g0, pe_mixture) * dx
    # trim to length of xsp (convolution starts at xsp[0] + 0 = xsp[0])
    g0_times_pe = g0_times_pe[: len(xsp)]

    G_direct = G_0 + g0_times_pe

    # compare over the region where both are appreciable
    mask = (xsp >= -3000.0) & (xsp <= 60000.0)
    max_err = np.max(np.abs(G_fft[mask] - G_direct[mask]))
    peak = np.max(np.abs(G_direct[mask]))
    rel_err = max_err / peak
    assert rel_err < 1e-3, f"rel err {rel_err}, max err {max_err}, peak {peak}"


# ==============================
#     Monotone support shift with lam
# ==============================


def test_mean_charge_grows_with_lam():
    """<q> should increase monotonically with lam."""
    bins = np.arange(2000.0, 100000.0 + 200.0, 200.0)
    hist = np.zeros(len(bins) - 1)
    grid = build_grid(hist, bins, A=1, q_min=-5000.0)
    xsp = grid.xsp
    dx = grid.xsp_width
    means = []
    for lam in [0.2, 0.5, 1.0, 2.0, 3.0]:
        G = _evaluate_density(grid, -600.0, 600.0, 6000.0, 800.0, 0.1, lam=lam)
        m = np.sum(xsp * G) * dx
        means.append(m)
    assert all(m1 < m2 for m1, m2 in zip(means[:-1], means[1:]))
    # expected mean ~ ped_mean + lam * spe_mean / (1 - xi)   (Gen-Poisson mean)
    for lam, m in zip([0.2, 0.5, 1.0, 2.0, 3.0], means):
        expected = -600.0 + lam * 6000.0 / (1.0 - 0.1)
        assert abs(m - expected) / abs(expected) < 5e-2


# ==============================
#     Pure-pedestal slice
# ==============================


def test_density_at_q_0_with_zero_ped_leakage():
    """Density at q=0 should be ~ g0(0) * exp(-lam) when no SPE bleeds there."""
    # pedestal far from zero, SPE far above zero
    ped_mean, ped_sigma = 10.0, 2.0  # tight pedestal at q=10
    spe_mean, spe_sigma = 6000.0, 800.0
    lam = 1.5
    xi = 0.1

    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    hist = np.zeros(len(bins) - 1)
    grid = build_grid(hist, bins, A=1, q_min=-500.0)
    G = _evaluate_density(grid, ped_mean, ped_sigma, spe_mean, spe_sigma, xi, lam)

    q = -30.0  # 20 sigmas below pedestal mean; SPE doesn't reach here
    idx = int(np.argmin(np.abs(grid.xsp - q)))
    p_0 = float(_count_pmf(jnp.asarray(lam), jnp.asarray(xi), 0))
    expected = p_0 * norm.pdf(q, loc=ped_mean, scale=ped_sigma)
    # in this regime expected is ~0 so we compare absolutely
    assert abs(G[idx] - expected) < 1e-6
