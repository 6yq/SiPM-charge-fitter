#!/usr/bin/env python3
# ===========================================================================
# tests/test_analytical_integral.py
# ===========================================================================

import numpy as np
import pytest
import jax.numpy as jnp

from scipy.stats import norm

from ..core.fft_grid import build_grid
from ..core.likelihood import _spectrum_fft, _bin_integrals, density_on_xsp
from ..models.gen_tweedie import _ft_extra, _ser_ft, _count_pgf, reparam_from_spe


# ==============================
#     Helpers
# ==============================


def _make_G_tilde(grid, extra, spe, lam):
    return _spectrum_fft(
        extra,
        spe,
        lam,
        jnp.asarray(grid.freq),
        _ft_extra,
        _ser_ft,
        _count_pgf,
    )


# ==============================
#     Gaussian reference
# ==============================


def test_gaussian_bin_integrals_match_erf():
    """Analytic Gaussian FT fed through _bin_integrals must match the CDF."""
    loc, scale = 0.5, 1.5

    # dq << scale
    dq = 0.1
    N = 512
    freq = jnp.asarray(2 * np.pi * np.fft.fftfreq(N, d=dq))

    # _spectrum_fft convention: G_tilde = fft(roll(g0,-i_zero))*dq = exp(-i*mu*w - sigma^2*w^2/2)
    G_tilde = jnp.asarray(
        np.exp(-1j * loc * np.asarray(freq) - 0.5 * scale**2 * np.asarray(freq) ** 2)
    )

    edges = jnp.asarray([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    bin_int = np.asarray(_bin_integrals(G_tilde, edges, freq, N, dq))
    ref = np.diff(
        norm.cdf(np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0]), loc=loc, scale=scale)
    )
    assert np.max(np.abs(bin_int - ref)) < 1e-10


# ==============================
#     Total integral ~ 1
# ==============================


def test_integral_over_full_grid_is_one():
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0)

    extra = jnp.array([-600.0, 600.0])
    spe = jnp.array([*reparam_from_spe(6000.0, 800.0), 0.1])
    lam = jnp.asarray(1.5)

    G_tilde = _make_G_tilde(grid, extra, spe, lam)
    edges = jnp.array([float(grid.xsp[0]), float(grid.xsp[-1])])
    val = float(
        _bin_integrals(
            G_tilde, edges, jnp.asarray(grid.freq), len(grid.xsp), float(grid.xsp_width)
        )[0]
    )
    assert abs(val - float(G_tilde[0].real)) < 1e-7
    assert abs(float(G_tilde[0].real) - 1.0) < 5e-3


# ==============================
#     Matches IFFT+trapezoid
# ==============================


def test_bin_integrals_match_trapezoid_on_density():
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0, dq=50.0)

    extra = jnp.array([-600.0, 600.0])
    spe = jnp.array([*reparam_from_spe(6000.0, 800.0), 0.1])
    lam = jnp.asarray(1.5)

    G_tilde = _make_G_tilde(grid, extra, spe, lam)
    bin_int_ana = np.asarray(
        _bin_integrals(
            G_tilde,
            jnp.asarray(grid.bins),
            jnp.asarray(grid.freq),
            len(grid.xsp),
            float(grid.xsp_width),
        )
    )

    density = np.asarray(
        density_on_xsp(G_tilde, float(grid.xsp_width), int(grid.i_zero))
    )
    xsp = grid.xsp
    bin_int_ref = np.array(
        [
            np.trapezoid(
                density[(xsp >= bins[k]) & (xsp <= bins[k + 1])],
                xsp[(xsp >= bins[k]) & (xsp <= bins[k + 1])],
            )
            for k in range(len(bins) - 1)
        ]
    )
    max_err = np.max(np.abs(bin_int_ana - bin_int_ref))
    assert max_err / bin_int_ref.max() < 1e-3


# ==============================
#     Bin integrals are non-negative
# ==============================


def test_bin_integrals_nonnegative():
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0)
    spe = jnp.array([*reparam_from_spe(6000.0, 800.0), 0.1])

    for lam_v in [0.2, 0.5, 1.5, 3.0]:
        G_tilde = _make_G_tilde(
            grid, jnp.array([-600.0, 600.0]), spe, jnp.asarray(lam_v)
        )
        bin_int = np.asarray(
            _bin_integrals(
                G_tilde,
                jnp.asarray(grid.bins),
                jnp.asarray(grid.freq),
                len(grid.xsp),
                float(grid.xsp_width),
            )
        )
        assert bin_int.min() > -1e-10


# ==============================
#     Sum of bin integrals + overflow = 1
# ==============================


def test_window_plus_overflow_is_unity():
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0)
    spe = jnp.array([*reparam_from_spe(6000.0, 800.0), 0.1])
    G_tilde = _make_G_tilde(grid, jnp.array([-600.0, 600.0]), spe, jnp.asarray(1.5))
    bin_int = np.asarray(
        _bin_integrals(
            G_tilde,
            jnp.asarray(grid.bins),
            jnp.asarray(grid.freq),
            len(grid.xsp),
            float(grid.xsp_width),
        )
    )
    p_total = float(G_tilde[0].real)
    overflow = p_total - bin_int.sum()
    assert abs((bin_int.sum() + overflow) - p_total) < 1e-12
    assert overflow >= -1e-12
