# ===========================================================================
# tests/test_gen_tweedie_fit.py
#
# End-to-end recovery tests: fit toy MC and verify that MLE parameters
# match the generating truth within statistical tolerance.
# ===========================================================================

import numpy as np
import pytest

from math import log

from ..models.gen_tweedie import GenTweedieFitter
from .conftest import (
    PED_MEAN,
    PED_SIGMA,
    SPE_MEAN,
    SPE_SIGMA,
    Q_MIN,
    N_EVENTS,
)


# ==============================
#     Helpers
# ==============================


def _make_fitter(mc):
    return GenTweedieFitter(
        hist=mc["hist"],
        bins=mc["bins"],
        A=N_EVENTS,
        q_min=Q_MIN,
        lam_init=mc["lam"] * 0.9,  # slightly off
    )


def _assert_close(name, fitted, true, rtol):
    rel = abs(fitted - true) / max(abs(true), 1e-9)
    assert rel < rtol, f"{name}: fitted={fitted}, true={true}, rel_err={rel}"


# ==============================
#     Low-lam recovery
# ==============================


def test_fit_recovers_low_lam(toy_mc_low_lam):
    fit = _make_fitter(toy_mc_low_lam)
    result = fit.fit_mle()

    assert result.converged

    log_A = result.theta[fit.layout["log_A"]][0]
    extra = result.theta[fit.layout["extra"]]
    spe = result.theta[fit.layout["spe"]]
    lam_hat = result.theta[fit.layout["lam"]][0]

    report = fit.spe_report(spe)

    _assert_close("log_A", log_A, log(N_EVENTS), rtol=5e-3)
    _assert_close("spe_mean", report["spe_mean"], SPE_MEAN, rtol=5e-2)
    _assert_close("spe_sigma", report["spe_sigma"], SPE_SIGMA, rtol=1e-1)
    _assert_close("lam", lam_hat, toy_mc_low_lam["lam"], rtol=5e-2)
    # xi is notoriously hard at low lam; loose tolerance
    assert 0.0 < report["xi"] < 0.5


def test_fit_recovers_mid_lam(toy_mc_mid_lam):
    fit = _make_fitter(toy_mc_mid_lam)
    result = fit.fit_mle()
    assert result.converged

    spe = result.theta[fit.layout["spe"]]
    lam_hat = result.theta[fit.layout["lam"]][0]
    report = fit.spe_report(spe)

    _assert_close("spe_mean", report["spe_mean"], SPE_MEAN, rtol=3e-2)
    _assert_close("spe_sigma", report["spe_sigma"], SPE_SIGMA, rtol=5e-2)
    _assert_close("lam", lam_hat, toy_mc_mid_lam["lam"], rtol=3e-2)
    _assert_close("xi", report["xi"], toy_mc_mid_lam["xi"], rtol=5e-1)


def test_fit_recovers_high_lam(toy_mc_high_lam):
    fit = _make_fitter(toy_mc_high_lam)
    result = fit.fit_mle()
    assert result.converged

    spe = result.theta[fit.layout["spe"]]
    lam_hat = result.theta[fit.layout["lam"]][0]
    report = fit.spe_report(spe)

    _assert_close("spe_mean", report["spe_mean"], SPE_MEAN, rtol=3e-2)
    _assert_close("spe_sigma", report["spe_sigma"], SPE_SIGMA, rtol=5e-2)
    _assert_close("lam", lam_hat, toy_mc_high_lam["lam"], rtol=3e-2)
    _assert_close("xi", report["xi"], toy_mc_high_lam["xi"], rtol=3e-1)


# ==============================
#     Ped params get constrained
# ==============================


def test_pedestal_constrained_by_tail_leakage(toy_mc_mid_lam):
    """With ped_sigma=600, the upper tail leaks into the histogram starting
    at q=2000 (~4.3 sigma), so the pedestal params should be somewhat
    constrained — but expect loose tolerances."""
    fit = _make_fitter(toy_mc_mid_lam)
    result = fit.fit_mle()
    extra = result.theta[fit.layout["extra"]]
    # very loose: pedestal parameters should not run to the bounds
    pm_lo, pm_hi = fit.bounds[fit.layout["extra"].start]
    ps_lo, ps_hi = fit.bounds[fit.layout["extra"].start + 1]
    assert pm_lo + 10 < extra[0] < pm_hi - 10
    assert ps_lo + 10 < extra[1] < ps_hi - 10


# ==============================
#     Components integrate to bin totals
# ==============================


def test_component_counts_sum_to_total(toy_mc_mid_lam):
    """Sum_n estimate_component_counts(theta, n)  should equal the total
    predicted bin counts (summed over enough n)."""
    fit = _make_fitter(toy_mc_mid_lam)
    result = fit.fit_mle()
    total = fit.estimate_bin_counts(result.theta)
    # sum first ~15 PE contributions
    acc = np.zeros_like(total)
    for n in range(15):
        acc += fit.estimate_component_counts(result.theta, n)
    max_rel = np.max(np.abs(total - acc)) / max(total.max(), 1e-9)
    assert max_rel < 1e-3, f"component sum mismatch: max_rel={max_rel}"


# ==============================
#     Errors are finite
# ==============================


def test_fit_errors_finite(toy_mc_mid_lam):
    fit = _make_fitter(toy_mc_mid_lam)
    result = fit.fit_mle()
    # log_A, pedestal, SPE, lam — all should have finite errors
    assert np.all(np.isfinite(result.theta_err[fit.layout["log_A"]]))
    assert np.all(np.isfinite(result.theta_err[fit.layout["spe"]]))
    assert np.all(np.isfinite(result.theta_err[fit.layout["lam"]]))
