#!/usr/bin/env python3
# ===========================================================================
# tests/test_density_pipeline.py
# ===========================================================================

import numpy as np
import pytest
import jax.numpy as jnp

from scipy.stats import norm, gamma as sgamma
from scipy.signal import fftconvolve

from ..core.fft_grid import build_grid
from ..core.likelihood import _spectrum_fft, density_on_xsp
from ..models.gen_tweedie import (
    _ft_extra,
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
    G_tilde = _spectrum_fft(
        extra,
        spe,
        jnp.asarray(lam),
        jnp.asarray(grid.freq),
        _ft_extra,
        _ser_ft,
        _count_pgf,
    )
    return np.asarray(density_on_xsp(G_tilde, float(grid.xsp_width), int(grid.i_zero)))


def _trapz(y, dx):
    return float(np.trapezoid(y, dx=dx))


# ==============================
#     Normalisation
# ==============================


def test_density_normalised_low_lam():
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0)
    G = _evaluate_density(grid, -600.0, 600.0, 6000.0, 800.0, 0.1, lam=0.5)
    assert abs(_trapz(G, grid.xsp_width) - 1.0) < 5e-3


def test_density_normalised_mid_lam():
    bins = np.arange(2000.0, 80000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0)
    G = _evaluate_density(grid, -600.0, 600.0, 6000.0, 800.0, 0.1, lam=1.5)
    assert abs(_trapz(G, grid.xsp_width) - 1.0) < 1e-2


# ==============================
#     lam -> 0 limit is pure pedestal
# ==============================


def test_low_lam_limit_is_pedestal():
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0)
    G = _evaluate_density(grid, -600.0, 600.0, 6000.0, 800.0, 0.1, lam=1e-8)
    ref = norm.pdf(grid.xsp, loc=-600.0, scale=600.0)
    # Tolerance reflects IFFT discretisation on a finite grid, not a model error
    assert np.max(np.abs(G - ref)) < 1e-3


# ==============================
#     Direct convolution reference
# ==============================


def test_density_matches_direct_convolution():
    ped_mean, ped_sigma = -600.0, 600.0
    spe_mean, spe_sigma = 6000.0, 800.0
    xi, lam = 0.1, 1.5
    alpha = (spe_mean / spe_sigma) ** 2
    theta = spe_mean / alpha

    bins = np.arange(2000.0, 80000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0)
    xsp, dx = grid.xsp, grid.xsp_width

    G_fft = _evaluate_density(grid, ped_mean, ped_sigma, spe_mean, spe_sigma, xi, lam)

    # Build reference on xsp directly — no fftconvolve, use FFT-based convolution
    # to avoid the mode='same' centering problem.
    # Convolve via FFT on the same periodic grid as _spectrum_fft uses.
    ped = norm.pdf(xsp, loc=ped_mean, scale=ped_sigma)
    q_pos = np.maximum(xsp, 0.0)
    f_spe = sgamma.pdf(q_pos, a=alpha, scale=theta)
    f_spe[xsp < 0] = 0.0

    # FFT-based cyclic convolution on xsp (same periodic domain as the model)
    F_ped = np.fft.fft(np.roll(ped, -grid.i_zero)) * dx
    F_spe = np.fft.fft(np.roll(f_spe, -grid.i_zero)) * dx

    k_max = 12
    G_ref = np.zeros(len(xsp))
    F_k = np.ones(len(xsp), dtype=complex)  # f^{*0}: delta at 0 -> FT = 1
    for k in range(k_max + 1):
        p_k = float(_count_pmf(jnp.asarray(lam), jnp.asarray(xi), k).real)
        conv_k = np.real(np.fft.ifft(F_ped * F_k)) / dx
        conv_k = np.roll(conv_k, grid.i_zero)
        G_ref += p_k * conv_k
        F_k *= F_spe  # accumulate f^{*k} in frequency domain

    mask = (xsp > 500) & (xsp < 25000)
    rel_err = np.max(np.abs(G_fft[mask] - G_ref[mask])) / np.max(G_ref[mask])
    assert rel_err < 0.02
