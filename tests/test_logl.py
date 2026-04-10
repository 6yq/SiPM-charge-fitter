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

import jax
import numpy as np
import jax.numpy as jnp

from math import log
from dataclasses import replace

from ..core.fft_grid import build_grid
from ..core.likelihood import (
    make_binned_logl,
    make_unbinned_logl,
    make_lam_logl,
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


# =======================================
#     Consistency with density_on_xsp
# =======================================
BIN_WIDTH_RECON = BIN_WIDTH / 2


def _make_grid():
    bins = np.arange(HIST_LO, HIST_HI + BIN_WIDTH_RECON, BIN_WIDTH_RECON)
    grid = build_grid(
        np.zeros(len(bins) - 1), bins, A=1, q_min=Q_MIN, dq=BIN_WIDTH_RECON
    )
    return grid


def _truth_arrays(n_ch=1):
    a, b = reparam_from_spe(SPE_MEAN, SPE_SIGMA)
    extra = jnp.tile(jnp.array([PED_MEAN, PED_SIGMA]), (n_ch, 1))
    spe = jnp.tile(jnp.array([a, b, 0.1]), (n_ch, 1))
    return extra, spe


def _make_fns(grid):
    return make_lam_logl(
        jnp.asarray(grid.freq),
        float(grid.xsp_width),
        len(grid.xsp),
        _ft_extra,
        _ser_ft,
        _count_pgf,
    )


def test_logl_matches_density_on_xsp():
    """logl_and_grad_fn must agree with density_on_xsp to numerical precision."""
    grid = _make_grid()
    precompute_channels_fn, precompute_event_fn, logl_and_grad_fn = _make_fns(grid)

    a, b = reparam_from_spe(SPE_MEAN, SPE_SIGMA)
    extra1 = jnp.array([[PED_MEAN, PED_SIGMA]])
    spe1 = jnp.array([[a, b, 0.1]])
    lam1 = jnp.array([1.5])
    Q1 = jnp.array([8000.0], dtype=jnp.float32)
    fired_idx = jnp.array([0], dtype=jnp.int32)

    # Reference via density_on_xsp
    G_tilde = _spectrum_fft(
        extra1[0],
        spe1[0],
        lam1[0],
        jnp.asarray(grid.freq),
        _ft_extra,
        _ser_ft,
        _count_pgf,
    )
    density = density_on_xsp(G_tilde, float(grid.xsp_width), int(grid.i_zero))
    # Interpolate at Q=8000 to get reference G(Q)
    xsp = grid.xsp
    ref_G = float(jnp.interp(jnp.array(8000.0), jnp.asarray(xsp), density))
    ref_logl = float(jnp.log(jnp.maximum(ref_G, 1e-32)))

    s_all, g0_all, u_all = precompute_channels_fn(extra1, spe1)
    u_sel, g0_phase = precompute_event_fn(Q1, fired_idx, g0_all, u_all)
    ll, _ = logl_and_grad_fn(lam1, u_sel, g0_phase)

    # Tolerance reflects interpolation vs exact dot-product difference
    assert abs(float(ll[0]) - ref_logl) < 0.1


# ==========================================
#     Analytic gradient matches jax.grad
# ==========================================


def test_analytic_grad_matches_jax_grad():
    """d(logl)/d(lam) from analytic formula must match jax.grad."""
    grid = _make_grid()
    precompute_channels_fn, precompute_event_fn, logl_and_grad_fn = _make_fns(grid)

    extra, spe = _truth_arrays(n_ch=10)
    lam = jnp.ones(10) * 1.5
    Q = jnp.asarray(
        np.random.default_rng(0).uniform(2000, 50000, 10), dtype=jnp.float32
    )
    fired_idx = jnp.arange(10, dtype=jnp.int32)

    s_all, g0_all, u_all = precompute_channels_fn(extra, spe)
    u_sel, g0_phase = precompute_event_fn(Q, fired_idx, g0_all, u_all)
    _, grad_analytic = logl_and_grad_fn(lam, u_sel, g0_phase)

    grad_auto = jax.grad(
        lambda lam: jnp.sum(logl_and_grad_fn(lam, u_sel, g0_phase)[0])
    )(lam)

    assert np.max(np.abs(np.asarray(grad_analytic - grad_auto))) < 1e-8


# =========================================
#     Finite values at realistic params
# =========================================


def test_logl_and_grad_finite():
    grid = _make_grid()
    precompute_channels_fn, precompute_event_fn, logl_and_grad_fn = _make_fns(grid)

    rng = np.random.default_rng(1)
    n_ch = 300
    extra, spe = _truth_arrays(n_ch)
    lam = jnp.asarray(rng.uniform(0.1, 3.0, n_ch))
    Q = jnp.asarray(rng.uniform(2000, 50000, n_ch), dtype=jnp.float32)
    fired_idx = jnp.arange(n_ch, dtype=jnp.int32)

    s_all, g0_all, u_all = precompute_channels_fn(extra, spe)
    u_sel, g0_phase = precompute_event_fn(Q, fired_idx, g0_all, u_all)
    ll, grad = logl_and_grad_fn(lam, u_sel, g0_phase)

    assert np.all(np.isfinite(np.asarray(ll)))
    assert np.all(np.isfinite(np.asarray(grad)))


# ========================
#     Local optimality
# ========================


def test_logl_higher_at_truth_lam(toy_mc_mid_lam):
    """logl at truth lam must exceed logl at a perturbed lam."""
    grid = _make_grid()
    precompute_channels_fn, precompute_event_fn, logl_and_grad_fn = _make_fns(grid)

    charges = toy_mc_mid_lam["charges"]
    lam_true = float(toy_mc_mid_lam["lam"])
    extra, spe = _truth_arrays(n_ch=1)

    # Use a single charge drawn near the SPE peak
    Q = jnp.array(
        [float(charges[np.argmin(np.abs(charges - SPE_MEAN))])], dtype=jnp.float32
    )
    fired_idx = jnp.array([0], dtype=jnp.int32)

    s_all, g0_all, u_all = precompute_channels_fn(extra, spe)
    u_sel, g0_phase = precompute_event_fn(Q, fired_idx, g0_all, u_all)

    ll_true, _ = logl_and_grad_fn(jnp.array([lam_true]), u_sel, g0_phase)
    ll_perturbed, _ = logl_and_grad_fn(jnp.array([lam_true * 1.5]), u_sel, g0_phase)

    assert float(ll_true[0]) > float(ll_perturbed[0])


# ====================================
#     Precompute invariance to lam
# ====================================


def test_precompute_independent_of_lam():
    """Channel/event precompute outputs must not depend on lam."""
    grid = _make_grid()
    precompute_channels_fn, precompute_event_fn, _ = _make_fns(grid)

    extra, spe = _truth_arrays(n_ch=5)
    Q = jnp.asarray([5000.0, 8000.0, 12000.0, 20000.0, 35000.0], dtype=jnp.float32)
    fired_idx = jnp.arange(5, dtype=jnp.int32)

    s1, g01, u1 = precompute_channels_fn(extra, spe)
    s2, g02, u2 = precompute_channels_fn(extra, spe)

    us1, gp1 = precompute_event_fn(Q, fired_idx, g01, u1)
    us2, gp2 = precompute_event_fn(Q, fired_idx, g02, u2)

    assert jnp.allclose(s1, s2)
    assert jnp.allclose(g01, g02)
    assert jnp.allclose(u1, u2)
    assert jnp.allclose(us1, us2)
    assert jnp.allclose(gp1, gp2)


# ==============================
#     Performance smoke test
# ==============================


def test_performance_300_fired_channels():
    """logl+grad for 300 fired channels must complete in < 50ms after warmup."""
    import time

    grid = _make_grid()
    precompute_channels_fn, precompute_event_fn, logl_and_grad_fn = _make_fns(grid)

    rng = np.random.default_rng(2)
    n_ch = 300
    extra, spe = _truth_arrays(n_ch)
    lam = jnp.asarray(rng.uniform(0.1, 3.0, n_ch))
    Q = jnp.asarray(rng.uniform(2000, 50000, n_ch), dtype=jnp.float32)
    fired_idx = jnp.arange(n_ch, dtype=jnp.int32)

    s_all, g0_all, u_all = precompute_channels_fn(extra, spe)
    u_sel, g0_phase = precompute_event_fn(Q, fired_idx, g0_all, u_all)

    # Warmup
    logl_and_grad_fn(lam, u_sel, g0_phase)[0].block_until_ready()

    t0 = time.time()
    for _ in range(20):
        ll, g = logl_and_grad_fn(lam, u_sel, g0_phase)
        ll.block_until_ready()
    elapsed_ms = (time.time() - t0) / 20 * 1000

    assert elapsed_ms < 50.0, f"logl+grad took {elapsed_ms:.1f}ms, expected < 50ms"
