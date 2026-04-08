#!/usr/bin/env python3
# ===========================================================================
# tests/test_logl.py
#
# Log-likelihood tests for both make_binned_logl and make_unbinned_logl.
#
# Binned invariants:
#   1. Finite value and finite gradients at truth on noisy toy MC.
#   2. Gradient vanishes at truth on a noise-free model-built histogram.
#   3. Continuous dependence on log_A.
#   4. Overflow mass varies with lam.
#
# Unbinned invariants:
#   1. Finite value and finite gradients at truth on noisy toy MC.
#   2. logL is higher at truth than at a perturbed point (local optimality).
#   3. Continuous dependence on log_A.
#   4. Normalised score is small at truth on model-drawn data.
# ===========================================================================

import numpy as np
import jax
import jax.numpy as jnp

from math import log
from dataclasses import replace

from ..core.fft_grid import build_grid
from ..core.likelihood import (
    make_binned_logl,
    make_unbinned_logl,
    _spectrum_fft,
    _bin_integrals,
    density_on_xsp,
)
from ..models.gen_tweedie import _ft_extra, _ser_ft, _count_pgf, reparam_from_spe

from .conftest import (
    PED_MEAN,
    PED_SIGMA,
    SPE_MEAN,
    SPE_SIGMA,
    Q_MIN,
    HIST_LO,
    HIST_HI,
    BIN_WIDTH,
    N_EVENTS,
)


# ======================
#     Shared helpers
# ======================


def _truth_theta(lam, xi):
    a, b = reparam_from_spe(SPE_MEAN, SPE_SIGMA)
    return dict(
        log_A=jnp.asarray(log(N_EVENTS)),
        extra=jnp.array([PED_MEAN, PED_SIGMA]),
        spe=jnp.array([a, b, xi]),
        lam=jnp.asarray(lam),
    )


def _G_tilde(grid, t):
    return _spectrum_fft(
        t["extra"],
        t["spe"],
        t["lam"],
        jnp.asarray(grid.freq),
        _ft_extra,
        _ser_ft,
        _count_pgf,
    )


def _build_grid_from_charges(charges):
    bins = np.arange(HIST_LO, HIST_HI + BIN_WIDTH, BIN_WIDTH)
    hist, _ = np.histogram(charges, bins=bins)
    return build_grid(hist, bins, A=N_EVENTS, q_min=Q_MIN)


def _sample_from_model_density(grid, t, n_samples, seed=42):
    """Inverse-CDF sampling from the model's IFFT density."""
    G_tilde = _G_tilde(grid, t)
    density = np.asarray(
        density_on_xsp(G_tilde, float(grid.xsp_width), int(grid.i_zero))
    )
    density = np.maximum(density, 0.0)
    density /= density.sum() * grid.xsp_width
    cdf = np.cumsum(density) * grid.xsp_width
    cdf /= cdf[-1]
    u = np.random.default_rng(seed).uniform(size=n_samples)
    return jnp.asarray(np.interp(u, cdf, grid.xsp), dtype=jnp.float32)


# =============================
#     Binned: finite values
# =============================


def test_binned_logl_finite_at_truth(toy_mc_low_lam):
    grid = _build_grid_from_charges(toy_mc_low_lam["charges"])
    logl = make_binned_logl(grid, _ft_extra, _ser_ft, _count_pgf)
    t = _truth_theta(toy_mc_low_lam["lam"], toy_mc_low_lam["xi"])
    assert np.isfinite(float(logl(t["log_A"], t["extra"], t["spe"], t["lam"])))


def test_binned_logl_gradient_finite(toy_mc_mid_lam):
    grid = _build_grid_from_charges(toy_mc_mid_lam["charges"])
    logl = make_binned_logl(grid, _ft_extra, _ser_ft, _count_pgf)
    t = _truth_theta(toy_mc_mid_lam["lam"], toy_mc_mid_lam["xi"])
    g = jax.grad(logl, argnums=(0, 1, 2, 3))(t["log_A"], t["extra"], t["spe"], t["lam"])
    for gi in g:
        assert np.all(np.isfinite(np.asarray(gi)))


# =============================
#     Binned: gradient zero
# =============================


def test_binned_gradient_zero_on_model_histogram():
    """Histogram built directly from model bin integrals — gradient at truth
    must vanish to FFT roundoff."""
    bins = np.arange(HIST_LO, HIST_HI + BIN_WIDTH, BIN_WIDTH)
    grid0 = build_grid(np.zeros(len(bins) - 1), bins, A=N_EVENTS, q_min=Q_MIN)
    t = _truth_theta(lam=1.5, xi=0.1)

    G_tilde = _G_tilde(grid0, t)
    bin_int = np.asarray(
        _bin_integrals(
            G_tilde,
            jnp.asarray(grid0.bins),
            jnp.asarray(grid0.freq),
            len(grid0.xsp),
            float(grid0.xsp_width),
        )
    )
    A_now = float(np.exp(float(t["log_A"])))
    fake_hist = A_now * bin_int
    fake_zero = A_now * (float(G_tilde[0].real) - bin_int.sum())
    fake_A = int(round(fake_hist.sum() + fake_zero))

    grid = build_grid(fake_hist, bins, A=fake_A, q_min=Q_MIN)
    grid = replace(grid, zero=fake_zero)
    logl = make_binned_logl(grid, _ft_extra, _ser_ft, _count_pgf)
    theta = dict(t, log_A=jnp.asarray(log(fake_A)))

    g = jax.grad(lambda th: logl(th["log_A"], th["extra"], th["spe"], th["lam"]))(theta)
    assert abs(float(g["log_A"])) < 1e-4
    assert np.max(np.abs(np.asarray(g["extra"]))) < 1e-4
    assert np.max(np.abs(np.asarray(g["spe"]))) < 1e-4
    assert abs(float(g["lam"])) < 1e-4


# ===================================
#     Binned: continuous in log_A
# ===================================


def test_binned_logl_continuous_in_log_A(toy_mc_mid_lam):
    grid = _build_grid_from_charges(toy_mc_mid_lam["charges"])
    logl = make_binned_logl(grid, _ft_extra, _ser_ft, _count_pgf)
    t = _truth_theta(toy_mc_mid_lam["lam"], toy_mc_mid_lam["xi"])
    v1 = float(logl(t["log_A"], t["extra"], t["spe"], t["lam"]))
    v2 = float(logl(t["log_A"] + 0.001, t["extra"], t["spe"], t["lam"]))
    assert np.isfinite(v1) and np.isfinite(v2)
    assert abs(v1 - v2) > 1e-4


# ========================================
#     Binned: overflow varies with lam
# ========================================


def test_binned_overflow_varies_with_lam():
    bins = np.arange(HIST_LO, HIST_HI + BIN_WIDTH, BIN_WIDTH)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=N_EVENTS, q_min=Q_MIN)
    spe = jnp.array([*reparam_from_spe(SPE_MEAN, SPE_SIGMA), 0.1])

    overflows = []
    for lam_v in [0.5, 1.5, 3.0, 5.0]:
        t = dict(
            extra=jnp.array([PED_MEAN, PED_SIGMA]), spe=spe, lam=jnp.asarray(lam_v)
        )
        G_tilde = _G_tilde(grid, t)
        bin_int = np.asarray(
            _bin_integrals(
                G_tilde,
                jnp.asarray(grid.bins),
                jnp.asarray(grid.freq),
                len(grid.xsp),
                float(grid.xsp_width),
            )
        )
        overflows.append(float(G_tilde[0].real) - bin_int.sum())

    assert max(overflows) - min(overflows) > 1e-3


# ===============================
#     Unbinned: finite values
# ===============================


def test_unbinned_logl_finite_at_truth(toy_mc_low_lam):
    charges = toy_mc_low_lam["charges"]
    grid = _build_grid_from_charges(charges)
    logl = make_unbinned_logl(
        jnp.asarray(charges, dtype=jnp.float32), grid, _ft_extra, _ser_ft, _count_pgf
    )
    t = _truth_theta(toy_mc_low_lam["lam"], toy_mc_low_lam["xi"])
    assert np.isfinite(float(logl(t["log_A"], t["extra"], t["spe"], t["lam"])))


def test_unbinned_logl_gradient_finite(toy_mc_mid_lam):
    charges = toy_mc_mid_lam["charges"]
    grid = _build_grid_from_charges(charges)
    logl = make_unbinned_logl(
        jnp.asarray(charges, dtype=jnp.float32), grid, _ft_extra, _ser_ft, _count_pgf
    )
    t = _truth_theta(toy_mc_mid_lam["lam"], toy_mc_mid_lam["xi"])
    g = jax.grad(logl, argnums=(0, 1, 2, 3))(t["log_A"], t["extra"], t["spe"], t["lam"])
    for gi in g:
        assert np.all(np.isfinite(np.asarray(gi)))


# ==================================
#     Unbinned: local optimality
# ==================================


def test_unbinned_logl_higher_at_truth(toy_mc_high_lam):
    """logL at truth must exceed logL at a 20%-perturbed lam."""
    charges = toy_mc_high_lam["charges"]
    grid = _build_grid_from_charges(charges)
    logl = make_unbinned_logl(
        jnp.asarray(charges, dtype=jnp.float32), grid, _ft_extra, _ser_ft, _count_pgf
    )
    t = _truth_theta(toy_mc_high_lam["lam"], toy_mc_high_lam["xi"])
    ll_0 = float(logl(t["log_A"], t["extra"], t["spe"], t["lam"]))
    ll_p = float(logl(t["log_A"], t["extra"], t["spe"], t["lam"] * 1.2))
    assert ll_0 > ll_p


# =====================================
#     Unbinned: continuous in log_A
# =====================================


def test_unbinned_logl_continuous_in_log_A(toy_mc_mid_lam):
    charges = toy_mc_mid_lam["charges"]
    grid = _build_grid_from_charges(charges)
    logl = make_unbinned_logl(
        jnp.asarray(charges, dtype=jnp.float32), grid, _ft_extra, _ser_ft, _count_pgf
    )
    t = _truth_theta(toy_mc_mid_lam["lam"], toy_mc_mid_lam["xi"])
    v1 = float(logl(t["log_A"], t["extra"], t["spe"], t["lam"]))
    v2 = float(logl(t["log_A"] + 0.001, t["extra"], t["spe"], t["lam"]))
    assert np.isfinite(v1) and np.isfinite(v2)
    assert abs(v1 - v2) > 1e-4


# =================================================
#     Unbinned: small normalised score at truth
# =================================================


def test_unbinned_score_small_at_truth_on_model_sample():
    """Score at truth on model-drawn data must be O(sqrt(N)), i.e.
    score / N < 0.5 for all parameters."""
    t = _truth_theta(lam=1.5, xi=0.1)
    bins = np.arange(HIST_LO, HIST_HI + BIN_WIDTH, BIN_WIDTH)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=N_EVENTS, q_min=Q_MIN)

    Q = _sample_from_model_density(grid, t, n_samples=N_EVENTS)
    logl = make_unbinned_logl(Q, grid, _ft_extra, _ser_ft, _count_pgf)

    g = jax.grad(lambda th: logl(th["log_A"], th["extra"], th["spe"], th["lam"]))(t)
    assert abs(float(g["log_A"])) / N_EVENTS < 0.5
    assert np.max(np.abs(np.asarray(g["extra"]))) / N_EVENTS < 0.5
    assert np.max(np.abs(np.asarray(g["spe"]))) / N_EVENTS < 0.5
    assert abs(float(g["lam"])) / N_EVENTS < 0.5
