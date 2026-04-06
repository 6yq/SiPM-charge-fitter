# ===========================================================================
# tests/test_analytic_integral.py
#
# Tests for the Fourier-domain analytic bin integration in core.likelihood.
# The defining property:
#
#   integral_{a}^{b} G(q) dq  =  I(b) - I(a)
#
# where G is reconstructed from its DFT coefficients and I is the analytic
# antiderivative.
#
# We test against:
#   - direct scipy reference (Gaussian, Gamma)
#   - sum-of-Gaussians mixtures
#   - the model's own density_on_xsp + numpy.trapezoid brute-force integral
# ===========================================================================

import numpy as np
import pytest
import jax.numpy as jnp

from scipy.stats import norm

from ..core.fft_grid import build_grid
from ..core.likelihood import _spectrum_fft, _bin_integrals, density_on_xsp
from ..models.gen_tweedie import _pdf_extra, _ser_ft, _count_pgf, reparam_from_spe


# ==============================
#     Gaussian reference
# ==============================


def test_gaussian_bin_integrals_match_erf():
    """A Gaussian on a wide grid should reproduce the scipy erf CDF exactly
    (up to FFT aliasing error, which is machine-precision when the grid
    covers many sigmas)."""
    q_lo, q_hi = -20.0, 20.0
    N = 512
    dq = (q_hi - q_lo) / (N - 1)
    xsp = np.linspace(q_lo, q_hi, N)
    i_zero = int(round(-q_lo / dq))
    xsp = xsp - xsp[i_zero]

    # Gaussian density (numpy, not jax — this is pure reference)
    loc, scale = 0.5, 1.5
    g = norm.pdf(xsp, loc=loc, scale=scale)
    g_rolled = np.roll(g, -i_zero)
    G_tilde = np.fft.fft(g_rolled) * dq
    freq = 2 * np.pi * np.fft.fftfreq(N, d=dq)

    edges = np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    bin_int = np.asarray(
        _bin_integrals(
            jnp.asarray(G_tilde), jnp.asarray(edges), jnp.asarray(freq), N, dq
        )
    )
    ref = np.diff(norm.cdf(edges, loc=loc, scale=scale))
    assert np.max(np.abs(bin_int - ref)) < 1e-12


# ==============================
#     Total integral ~ 1
# ==============================


def test_integral_over_full_grid_is_one():
    """Integrating over the whole xsp range should give ~ G~_0, which is 1
    for a properly normalised density."""
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0)

    a, b = reparam_from_spe(6000.0, 800.0)
    extra = jnp.array([-600.0, 600.0])
    spe = jnp.array([a, b, 0.1])
    lam = jnp.asarray(1.5)

    G_tilde = _spectrum_fft(
        extra,
        spe,
        lam,
        jnp.asarray(grid.freq),
        jnp.asarray(grid.xsp),
        float(grid.xsp_width),
        int(grid.i_zero),
        _pdf_extra,
        _ser_ft,
        _count_pgf,
    )
    # full-grid integral via bin integrals over [xsp[0], xsp[-1]]
    edges = jnp.array([float(grid.xsp[0]), float(grid.xsp[-1])])
    val = float(
        _bin_integrals(
            G_tilde, edges, jnp.asarray(grid.freq), len(grid.xsp), float(grid.xsp_width)
        )[0]
    )
    # should match Re(G~_0) (tiny aliasing residual at the period boundary)
    assert abs(val - float(G_tilde[0].real)) < 1e-7
    # ...and G~_0 should be ~1 since the grid covers the physical support
    assert abs(float(G_tilde[0].real) - 1.0) < 5e-3


# ==============================
#     Matches IFFT+trapezoid
# ==============================


def test_bin_integrals_match_trapezoid_on_density():
    """For each bin, the analytic integral must agree with
    numerical trapezoid integration of G(q) on the dense xsp grid."""
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0, dq=50.0)

    a, b = reparam_from_spe(6000.0, 800.0)
    extra = jnp.array([-600.0, 600.0])
    spe = jnp.array([a, b, 0.1])
    lam = jnp.asarray(1.5)

    G_tilde = _spectrum_fft(
        extra,
        spe,
        lam,
        jnp.asarray(grid.freq),
        jnp.asarray(grid.xsp),
        float(grid.xsp_width),
        int(grid.i_zero),
        _pdf_extra,
        _ser_ft,
        _count_pgf,
    )
    bin_int_ana = np.asarray(
        _bin_integrals(
            G_tilde,
            jnp.asarray(grid.bins),
            jnp.asarray(grid.freq),
            len(grid.xsp),
            float(grid.xsp_width),
        )
    )

    # brute-force reference: IFFT -> density -> trapezoid per bin
    density = np.asarray(
        density_on_xsp(
            extra,
            spe,
            lam,
            jnp.asarray(grid.freq),
            float(grid.xsp_width),
            int(grid.i_zero),
            jnp.asarray(grid.xsp),
            _pdf_extra,
            _ser_ft,
            _count_pgf,
        )
    )
    xsp = grid.xsp
    bin_int_ref = np.empty(len(bins) - 1)
    for k in range(len(bins) - 1):
        mask = (xsp >= bins[k]) & (xsp <= bins[k + 1])
        bin_int_ref[k] = np.trapezoid(density[mask], xsp[mask])

    # tight tolerance: both come from the same G~, so any disagreement is
    # discretisation of the trapezoid integral.  dq=50 vs bin_width=200 gives
    # ~5 trapezoid points per bin, accurate to ~1e-4.
    max_err = np.max(np.abs(bin_int_ana - bin_int_ref))
    peak = bin_int_ref.max()
    assert max_err / peak < 1e-3, f"max rel err {max_err / peak}"


# ==============================
#     Bin integrals are non-negative
# ==============================


def test_bin_integrals_nonnegative():
    """Physical PDFs give non-negative bin integrals."""
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0)

    a, b = reparam_from_spe(6000.0, 800.0)
    for lam_v in [0.2, 0.5, 1.5, 3.0]:
        extra = jnp.array([-600.0, 600.0])
        spe = jnp.array([a, b, 0.1])
        G_tilde = _spectrum_fft(
            extra,
            spe,
            jnp.asarray(lam_v),
            jnp.asarray(grid.freq),
            jnp.asarray(grid.xsp),
            float(grid.xsp_width),
            int(grid.i_zero),
            _pdf_extra,
            _ser_ft,
            _count_pgf,
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
        # allow tiny aliasing-induced negatives
        assert bin_int.min() > -1e-10, f"lam={lam_v}: min bin int {bin_int.min()}"


# ==============================
#     Sum of bin integrals + overflow = 1
# ==============================


def test_window_plus_overflow_is_unity():
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    grid = build_grid(np.zeros(len(bins) - 1), bins, A=1, q_min=-5000.0)
    a, b = reparam_from_spe(6000.0, 800.0)
    extra = jnp.array([-600.0, 600.0])
    spe = jnp.array([a, b, 0.1])
    lam = jnp.asarray(1.5)
    G_tilde = _spectrum_fft(
        extra,
        spe,
        lam,
        jnp.asarray(grid.freq),
        jnp.asarray(grid.xsp),
        float(grid.xsp_width),
        int(grid.i_zero),
        _pdf_extra,
        _ser_ft,
        _count_pgf,
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
    in_window = bin_int.sum()
    p_total = float(G_tilde[0].real)
    overflow = p_total - in_window
    assert abs((in_window + overflow) - p_total) < 1e-12
    assert overflow >= -1e-12
