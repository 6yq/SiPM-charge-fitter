# ===========================================================================
# tests/conftest.py
#
# Shared fixtures for the unit-test suite.
#
# Realistic PMT parameters (as specified for the refactor):
#   pedestal : N(-600, 600^2)
#   SPE      : Gamma with mean=6000, sigma=800
#   q_min    : -5000 (xsp must cover the pedestal left tail)
#   hist_lo  : 2000 (histogram starts well above the pedestal)
# ===========================================================================

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)


# ==============================
#     Realistic constants
# ==============================


PED_MEAN = -600.0
PED_SIGMA = 600.0
SPE_MEAN = 6000.0
SPE_SIGMA = 800.0

Q_MIN = -5000.0
HIST_LO = 2000.0
HIST_HI = 60000.0
BIN_WIDTH = 200.0

N_EVENTS = 200_000
SEED = 1234


# ==============================
#     Gen-Poisson(lam, xi) sampler
# ==============================


def _gen_poisson_pmf(lam, xi, k_max=80):
    ks = np.arange(k_max + 1, dtype=float)
    log_p = np.empty_like(ks)
    log_p[0] = -lam
    for k in range(1, k_max + 1):
        log_p[k] = (
            np.log(lam)
            + (k - 1) * np.log(max(lam + xi * k, 1e-300))
            - lam
            - xi * k
            - float(np.sum(np.log(np.arange(1, k + 1))))
        )
    p = np.exp(log_p - log_p.max())
    p = np.maximum(p, 0.0)
    return p / p.sum()


def sample_gen_tweedie(
    n_events,
    lam,
    xi=0.1,
    ped_mean=PED_MEAN,
    ped_sigma=PED_SIGMA,
    spe_mean=SPE_MEAN,
    spe_sigma=SPE_SIGMA,
    seed=SEED,
):
    """Draw n_events charges from the compound Gen-Poisson-Gamma model."""
    rng = np.random.default_rng(seed)
    alpha = (spe_mean / spe_sigma) ** 2
    theta = spe_mean / alpha

    pmf = _gen_poisson_pmf(lam, xi)
    ks = rng.choice(len(pmf), size=n_events, p=pmf)

    charges = rng.normal(ped_mean, ped_sigma, size=n_events)
    nonzero = np.where(ks > 0)[0]
    total = int(ks[nonzero].sum())
    if total > 0:
        draws = rng.gamma(shape=alpha, scale=theta, size=total)
        idx = 0
        for i in nonzero:
            k = int(ks[i])
            charges[i] += draws[idx : idx + k].sum()
            idx += k

    return charges, ks


def build_hist(charges, lo=HIST_LO, hi=HIST_HI, width=BIN_WIDTH):
    bins = np.arange(lo, hi + width, width)
    hist, _ = np.histogram(charges, bins=bins)
    return hist, bins


# ==============================
#     Fixtures
# ==============================


@pytest.fixture(scope="session")
def realistic_params():
    return dict(
        ped_mean=PED_MEAN,
        ped_sigma=PED_SIGMA,
        spe_mean=SPE_MEAN,
        spe_sigma=SPE_SIGMA,
        q_min=Q_MIN,
        hist_lo=HIST_LO,
        hist_hi=HIST_HI,
        bin_width=BIN_WIDTH,
        n_events=N_EVENTS,
        seed=SEED,
    )


@pytest.fixture(scope="session")
def toy_mc_low_lam():
    """Low-light toy MC (lam=0.5)."""
    charges, ks = sample_gen_tweedie(N_EVENTS, lam=0.5, xi=0.1, seed=SEED)
    hist, bins = build_hist(charges)
    return dict(charges=charges, ks=ks, hist=hist, bins=bins, lam=0.5, xi=0.1)


@pytest.fixture(scope="session")
def toy_mc_mid_lam():
    """Mid-light toy MC (lam=1.5)."""
    charges, ks = sample_gen_tweedie(N_EVENTS, lam=1.5, xi=0.1, seed=SEED + 1)
    hist, bins = build_hist(charges)
    return dict(charges=charges, ks=ks, hist=hist, bins=bins, lam=1.5, xi=0.1)


@pytest.fixture(scope="session")
def toy_mc_high_lam():
    """High-light toy MC (lam=3.0)."""
    charges, ks = sample_gen_tweedie(N_EVENTS, lam=3.0, xi=0.1, seed=SEED + 2)
    hist, bins = build_hist(charges)
    return dict(charges=charges, ks=ks, hist=hist, bins=bins, lam=3.0, xi=0.1)
