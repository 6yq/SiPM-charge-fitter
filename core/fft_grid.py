# ===========================================================================
# core/fft_grid.py
#
# Minimal static FFT grid for one spectrum.
#
# After the switch to analytic Fourier-domain bin integration there is no
# longer a need for sub-sampling or Simpson weights: the spectrum density
# G(q) is evaluated implicitly from its Fourier coefficients G_tilde at
# arbitrary points via
#     G(q) = (1/L) * sum_n  G_tilde_n * exp(+i * w_n * q)
# and the bin integrals
#     Y_k = A * integral_{b_k}^{b_{k+1}} G(q) dq
# are computed directly in Fourier space (see core/likelihood.py).
#
# The grid here only needs to:
#   1. be uniform in q so we can FFT the pedestal g0,
#   2. cover the full physical support [q_min, q_max] of G so the periodic
#      image does not overlap with itself,
#   3. have a power-of-two length N for efficient FFT.
#
# No sub-sampling (xsp_width == bin_width unless overridden).
# ===========================================================================

import numpy as np

from math import log, exp
from dataclasses import dataclass


# ==============================
#     Grid container
# ==============================


@dataclass(frozen=True)
class FFTGrid:
    bins: np.ndarray  # histogram bin edges, len = nbin+1
    hist: np.ndarray  # bin counts, len = nbin
    A: int  # total events (including overflow)
    zero: int  # A - sum(hist)
    xsp: np.ndarray  # uniform grid for FFT, len = N (power of 2)
    xsp_width: float  # bin spacing of xsp
    i_zero: int  # index in xsp closest to q = 0
    freq: np.ndarray  # angular frequencies of xsp, len = N
    log_C: float  # log-factorial constant for Poisson log-L


# ==============================
#     Builder
# ==============================


def build_grid(
    hist,
    bins,
    A=None,
    q_min=None,
    q_max=None,
    dq=None,
    lam_init_hint=None,
):
    """Build the uniform FFT grid for one spectrum.

    Parameters
    ----------
    hist, bins : array-like
        Bin counts and edges.
    A : int or None
        Total events including overflow.  Defaults to sum(hist).
    q_min : float or None
        Left edge of the FFT grid.  Must sit below the pedestal left tail.
        Defaults to ``min(bins[0] - 5 * bin_width, 0.0)``.
    q_max : float or None
        Right edge of the FFT grid.  Must sit above the physical right
        support of G so the periodic image of the IFFT does not wrap.
        Defaults to ``bins[-1] + 5 * bin_width``.
    dq : float or None
        xsp spacing.  Defaults to the histogram bin_width (no sub-sampling).
    lam_init_hint : float or None
        Unused, kept for API compatibility.
    """
    hist = np.asarray(hist, dtype=float)
    bins = np.asarray(bins, dtype=float)
    A = int(A) if A is not None else int(hist.sum())
    zero = A - int(hist.sum())

    bin_width = float(bins[1] - bins[0])
    if dq is None:
        dq = bin_width

    # ==============================
    #     Choose grid extent
    # ==============================
    if q_min is None:
        q_min = float(min(bins[0] - 5 * bin_width, 0.0))
    if q_max is None:
        q_max = float(bins[-1] + 5 * bin_width)
    # always cover q = 0
    q_min = min(float(q_min), 0.0)
    q_max = max(float(q_max), 0.0)

    # ==============================
    #     Power-of-two length
    # ==============================
    span = q_max - q_min
    N = int(np.ceil(span / dq)) + 1

    xsp = np.linspace(q_min, q_min + (N - 1) * dq, num=N, endpoint=True)
    i_zero = int(round(-q_min / dq))
    # snap xsp so that xsp[i_zero] is exactly 0.0
    xsp = xsp - xsp[i_zero]

    freq = 2.0 * np.pi * np.fft.fftfreq(N, d=dq)

    # log-factorial constant for the extended-Poisson log-likelihood
    log_C = float(
        sum(np.sum(np.log(np.arange(1, int(n) + 1))) for n in hist)
        + np.sum(np.log(np.arange(1, zero + 1)))
    )

    return FFTGrid(
        bins=bins,
        hist=hist,
        A=A,
        zero=zero,
        xsp=xsp,
        xsp_width=dq,
        i_zero=i_zero,
        freq=freq,
        log_C=log_C,
    )
