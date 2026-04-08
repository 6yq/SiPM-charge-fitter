#!/usr/bin/env python3
# ====================================================================
# tests/test_gen_tweedie_fit.py
#
# End-to-end recovery tests: fit toy MC and verify that MLE parameters
# match the generating truth within statistical tolerance.
# ====================================================================

import numpy as np
import pytest

from math import log

from ..models.gen_tweedie import GenTweedieFitter
from .conftest import PED_MEAN, PED_SIGMA, SPE_MEAN, SPE_SIGMA, XI, N_EVENTS


# ===============
#     Helpers
# ===============


def _make_fitter(mc):
    return GenTweedieFitter(
        Q_raw=mc["charges"],
        lam_init=mc["lam"] * 0.9,
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
    spe = result.theta[fit.layout["spe"]]
    lam_hat = result.theta[fit.layout["lam"]][0]

    report = fit.spe_report(spe)

    _assert_close("log_A", log_A, log(N_EVENTS), rtol=5e-3)
    _assert_close("spe_mean", report["spe_mean"], SPE_MEAN, rtol=5e-2)
    _assert_close("spe_sigma", report["spe_sigma"], SPE_SIGMA, rtol=5e-2)
    _assert_close("lam", lam_hat, toy_mc_low_lam["lam"], rtol=1e-2)
    _assert_close("xi", report["xi"], XI, rtol=2e-2)


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
    _assert_close("xi", report["xi"], toy_mc_mid_lam["xi"], rtol=2e-2)


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
    _assert_close("xi", report["xi"], toy_mc_high_lam["xi"], rtol=2e-2)


# =======================================
#     Pedestal params get constrained
# =======================================


def test_pedestal_constrained_by_tail_leakage(toy_mc_mid_lam):
    fit = _make_fitter(toy_mc_mid_lam)
    result = fit.fit_mle()
    extra = result.theta[fit.layout["extra"]]
    pm_lo, pm_hi = fit.bounds[fit.layout["extra"].start]
    ps_lo, ps_hi = fit.bounds[fit.layout["extra"].start + 1]
    assert pm_lo + 10 < extra[0] < pm_hi - 10
    assert ps_lo + 10 < extra[1] < ps_hi - 10


# ==========================================
#     Components integrate to bin totals
# ==========================================


def test_component_counts_sum_to_total(toy_mc_mid_lam):
    fit = _make_fitter(toy_mc_mid_lam)
    result = fit.fit_mle()
    total = fit.estimate_bin_counts(result.theta)
    acc = sum(fit.estimate_component_counts(result.theta, n) for n in range(15))
    max_rel = np.max(np.abs(total - acc)) / max(total.max(), 1e-9)
    assert max_rel < 1e-3


# =========================
#     Errors are finite
# =========================


def test_fit_errors_finite(toy_mc_mid_lam):
    fit = _make_fitter(toy_mc_mid_lam)
    result = fit.fit_mle()
    assert np.all(np.isfinite(result.theta_err[fit.layout["log_A"]]))
    assert np.all(np.isfinite(result.theta_err[fit.layout["spe"]]))
    assert np.all(np.isfinite(result.theta_err[fit.layout["lam"]]))
