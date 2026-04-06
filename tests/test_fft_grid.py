# ===========================================================================
# tests/test_fft_grid.py
#
# Geometry invariants of the minimal FFT grid.
# ===========================================================================

import numpy as np
import pytest

from ..core.fft_grid import build_grid


def _make_hist():
    bins = np.arange(2000.0, 60000.0 + 200.0, 200.0)
    hist = np.full(len(bins) - 1, 100.0)
    return hist, bins


# ==============================
#     q=0 is always covered
# ==============================


def test_grid_always_reaches_zero():
    hist, bins = _make_hist()
    grid = build_grid(hist, bins, A=int(hist.sum()), q_min=500.0)
    assert grid.xsp[0] <= 0.0
    assert 0 <= grid.i_zero < len(grid.xsp)
    assert abs(grid.xsp[grid.i_zero]) < 1e-10  # snapped to exactly 0


def test_grid_respects_q_min():
    hist, bins = _make_hist()
    grid = build_grid(hist, bins, A=int(hist.sum()), q_min=-5000.0)
    assert grid.xsp[0] <= -5000.0 + grid.xsp_width


def test_grid_covers_q_max():
    hist, bins = _make_hist()
    grid = build_grid(hist, bins, A=int(hist.sum()), q_min=-5000.0, q_max=80000.0)
    assert grid.xsp[-1] >= 80000.0 - grid.xsp_width


# ==============================
#     Power-of-two length
# ==============================


def test_length_is_power_of_two():
    hist, bins = _make_hist()
    grid = build_grid(hist, bins, A=int(hist.sum()), q_min=-5000.0)
    N = len(grid.xsp)
    assert N & (N - 1) == 0, f"N={N} is not a power of 2"


# ==============================
#     Uniform spacing
# ==============================


def test_xsp_is_uniform():
    hist, bins = _make_hist()
    grid = build_grid(hist, bins, A=int(hist.sum()), q_min=-5000.0)
    diffs = np.diff(grid.xsp)
    assert np.max(np.abs(diffs - grid.xsp_width)) < 1e-9


def test_default_dq_is_bin_width():
    hist, bins = _make_hist()
    grid = build_grid(hist, bins, A=int(hist.sum()), q_min=-5000.0)
    assert np.isclose(grid.xsp_width, bins[1] - bins[0])


def test_custom_dq_honored():
    hist, bins = _make_hist()
    grid = build_grid(hist, bins, A=int(hist.sum()), q_min=-5000.0, dq=50.0)
    assert np.isclose(grid.xsp_width, 50.0)


# ==============================
#     Frequency axis
# ==============================


def test_freq_length_and_dc():
    hist, bins = _make_hist()
    grid = build_grid(hist, bins, A=int(hist.sum()), q_min=-5000.0)
    assert len(grid.freq) == len(grid.xsp)
    assert grid.freq[0] == 0.0  # DC term at index 0


# ==============================
#     Overflow bookkeeping
# ==============================


def test_zero_count_matches():
    hist, bins = _make_hist()
    A = int(hist.sum()) + 777
    grid = build_grid(hist, bins, A=A, q_min=-5000.0)
    assert grid.zero == 777


def test_log_C_finite():
    hist, bins = _make_hist()
    grid = build_grid(hist, bins, A=int(hist.sum()) + 500, q_min=-5000.0)
    assert np.isfinite(grid.log_C)
    assert grid.log_C > 0
