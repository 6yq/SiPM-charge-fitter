#!/usr/bin/env python3
import operator

import numpy as np
from typing import NamedTuple


# =============================================
#     Parameter bound / constraint checks
# =============================================


def isInBound(param, bound):
    """Return True if param lies within bound = (lo, hi), None means unbounded."""
    assert len(bound) == 2, "Bound must be a 2-tuple."
    lo, hi = bound
    if lo is not None and hi is not None:
        assert lo <= hi, "Lower bound must not exceed upper bound."
        return lo <= param <= hi
    if lo is None and hi is None:
        return True
    if lo is None:
        return param < hi
    return param > lo


def isParamsInBound(params, bounds):
    """Return True if every parameter lies within its bound."""
    for p, b in zip(params, bounds):
        if not isInBound(p, b):
            return False
    return True


def isParamsWithinConstraints(args, constraints):
    """Return True if args satisfy all linear constraints.

    Each constraint is a dict::

        {"coeffs": [(idx, coeff), ...], "threshold": rhs, "op": ">"}

    meaning  sum(coeff * args[idx])  op  rhs.
    """
    ops = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
    }
    for c in constraints:
        lhs = sum(coeff * args[idx] for idx, coeff in c["coeffs"])
        if not ops[c.get("op", ">")](lhs, c["threshold"]):
            return False
    return True


# ==============================
#     Numerical integration
# ==============================


def composite_simpson(pdf_slice, interval, sample):
    """Composite Simpson's rule over one bin.

    Parameters
    ----------
    pdf_slice : array-like, length sample+1
    interval : float
        Bin width.
    sample : int
        Number of sub-intervals (must be even).
    """
    result = pdf_slice[0] + pdf_slice[-1]
    result += 4 * sum(pdf_slice[i] for i in range(1, sample, 2))
    result += 2 * sum(pdf_slice[i] for i in range(2, sample, 2))
    return result * interval / 3


# ==============================
#     Bin merging
# ==============================


def merge_bins(hist, y, threshold=5):
    """Merge low-count bins toward the peak from both sides.

    Parameters
    ----------
    hist : ndarray   observed counts
    y    : ndarray   expected counts (same length)
    threshold : int  merge until all bins have counts above this value

    Returns
    -------
    hist_, y_ : ndarray
    """
    hist_ = hist.copy()
    y_ = y.copy()

    while True:
        peak_idx = np.argmax(hist_)
        low = np.where(hist_ <= threshold)[0]
        if low.size == 0:
            break

        idx = low[0]
        merged = False

        if idx < peak_idx:
            hist_ = np.append(
                np.append(hist_[:idx], hist_[idx] + hist_[idx + 1]), hist_[idx + 2 :]
            )
            y_ = np.append(np.append(y_[:idx], y_[idx] + y_[idx + 1]), y_[idx + 2 :])
            merged = True
        elif idx > peak_idx:
            idx = low[-1]
            hist_ = np.append(
                np.append(hist_[: idx - 1], hist_[idx - 1] + hist_[idx]),
                hist_[idx + 1 :],
            )
            y_ = np.append(
                np.append(y_[: idx - 1], y_[idx - 1] + y_[idx]), y_[idx + 1 :]
            )
            merged = True

        if not merged:
            break

    return hist_, y_


# ==============================
#     Extra-parameter registry
# ==============================


class ExtraParam(NamedTuple):
    name: str
    init: float
    bound: tuple  # (lo, hi); None means unbounded on that side


PEDESTAL_PARAMS: list[ExtraParam] = [
    ExtraParam("ped_mean", init=-600.0, bound=(-1200.0, 600.0)),
    ExtraParam("ped_sigma", init=200.0, bound=(1e-6, 1200.0)),
]

THRESHOLD_PARAMS: list[ExtraParam] = [
    ExtraParam("thres_center", init=2500, bound=(0.0, 6000.0)),
    ExtraParam("thres_scale", init=250.0, bound=(1e-6, 2000.0)),
]


# ==============================
#     Auto-initialisation
# ==============================


def compute_init(hist, edges):
    """Estimate pedestal and SPE initial parameters via a 4-Gaussian mixture fit
    using ROOT's TH1F + TF1 fitter.

    The 4 Gaussians correspond to n = 1, 2, 3, 4 PE::

        p(q) = sum_{n=1}^{4}  w_n * N(mu_0 + n*G,  sqrt(sigma_0^2 + n*sigma_G^2))

    Pedestal parameters are derived:
        ped_mean  = mu_0  = mu1 - G
        ped_sigma = sigma_0  (substituted with spe_sigma / sqrt(2))

    Parameters
    ----------
    hist  : ndarray   bin counts
    edges : ndarray   bin edges, length len(hist)+1

    Returns
    -------
    ped_mean  : float
    ped_sigma : float
    spe_mean  : float
    spe_sigma : float
    """
    import ROOT
    from scipy.signal import find_peaks

    ROOT.gErrorIgnoreLevel = ROOT.kError

    xs = (edges[:-1] + edges[1:]) / 2.0
    counts = hist.astype(float)
    if counts.sum() == 0:
        raise ValueError("Histogram is empty.")

    # robust gain estimate: use spacing between the two most prominent peaks
    # distance=3 bins minimum separation; prominence >= 1% of max to filter noise
    peaks, props = find_peaks(
        counts,
        prominence=counts.max() * 0.01,
        distance=max(3, len(counts) // 40),
    )
    if len(peaks) >= 2:
        # sort by position. channel 3621 has inusual prominence.
        top2 = peaks[:2]
        G_rough = float(xs[top2[1]] - xs[top2[0]])
        mu1_rough = float(xs[top2[0]])
    elif len(peaks) == 1:
        G_rough = float(xs[peaks[0]])
        mu1_rough = float(xs[peaks[0]])
    else:
        G_rough = float(np.average(xs, weights=counts))
        mu1_rough = G_rough

    # ==============================
    #     Build ROOT histogram
    # ==============================

    n_bins = len(hist)
    th1 = ROOT.TH1F("h_init", "", n_bins, edges[0], edges[-1])
    ROOT.SetOwnership(th1, False)
    for i, c in enumerate(counts):
        th1.SetBinContent(i + 1, c)
        th1.SetBinError(i + 1, max(c**0.5, 1.0))

    # ==============================
    #     Define 4-Gaussian TF1
    # ==============================
    # Parameters:
    #   [0] = ped_mean   (pedestal mean, directly)
    #   [1] = G          (gain = peak spacing)
    #   [2] = sigma_G    (SPE charge resolution)
    #   [3] = sigma_0    (pedestal sigma)
    #   [4] = w1, [5] = w2, [6] = w3   (weights; w4 = 1 - w1 - w2 - w3)
    #
    # n-th peak (n=1..4): mean = ped_mean + n*G,
    #                     sigma = sqrt(sigma_0^2 + n*sigma_G^2)
    # Amplitude = w_n * total * bin_width / (sqrt(2*pi) * sigma_n)

    total = float(counts.sum())
    bw = float(edges[1] - edges[0])

    # build formula; w4 expressed as (1 - w1 - w2 - w3)
    parts = []
    for n in range(1, 5):
        mean = f"([0]+{n}*[1])"
        sigma = f"sqrt([3]*[3]+{n}*[2]*[2])"
        w = f"[{3+n}]" if n < 4 else f"(1-[4]-[5]-[6])"
        amp = f"{total}*{bw}"
        parts.append(f"{w}*{amp}*exp(-0.5*((x-{mean})/{sigma})^2)/({sigma}*sqrt(2*pi))")
    formula = "+".join(parts)

    tf1 = ROOT.TF1("f_mix", formula, edges[0], edges[-1])
    ROOT.SetOwnership(tf1, False)
    tf1.SetNpx(n_bins * 4)

    ped_mean_rough = mu1_rough - G_rough

    tf1.SetParameters(
        ped_mean_rough, G_rough, 0.20 * G_rough, 0.05 * G_rough, 0.50, 0.25, 0.15
    )

    for i, name in enumerate(["ped_mean", "G", "sigma_G", "sigma_0", "w1", "w2", "w3"]):
        tf1.SetParName(i, name)

    tf1.SetParLimits(0, ped_mean_rough - 400, ped_mean_rough + 400)
    tf1.SetParLimits(1, G_rough * 0.3, G_rough * 2.0)
    tf1.SetParLimits(2, 0, 1e03)
    tf1.SetParLimits(3, 0, 1e04)
    tf1.SetParLimits(4, 0.01, 0.97)  # w1
    tf1.SetParLimits(5, 0.01, 0.97)  # w2
    tf1.SetParLimits(6, 0.01, 0.97)  # w3

    # ==============================
    #     Fit
    # ==============================

    # S: store result, R: use TF1 range, Q: quiet, B: respect limits
    th1.Fit("f_mix", "SRQB")

    ped_mean = float(tf1.GetParameter(0))
    G = float(tf1.GetParameter(1))
    sG = float(tf1.GetParameter(2))

    spe_mean = float(G)
    spe_sigma = float(sG)
    ped_sigma = spe_sigma / (2.0**0.5)

    print(f"[AUTO] ped_mean={ped_mean:.4g}  ped_sigma={ped_sigma:.4g}", flush=True)
    print(f"[AUTO] spe_mean={spe_mean:.4g}  spe_sigma={spe_sigma:.4g}", flush=True)

    return ped_mean, ped_sigma, spe_mean, spe_sigma


# ==============================
#     Chi-square functions
# ==============================


def merged_pearson_chi2(hist, y, zero, z, dof):
    """Pearson chi-square after merging low-count bins (threshold=5)."""
    y_reg, hist_reg = merge_bins(y, hist, threshold=5)
    ndf = len(hist_reg) - dof
    chi2 = float(np.sum((y_reg - hist_reg) ** 2 / y_reg) + (z - zero) ** 2 / z)
    return chi2, ndf


def modified_neyman_chi2_A(hist, y, zero, z, dof):
    """Modified Neyman chi-square A:  sum(y/n - n)."""
    ndf = len(hist) - dof
    mask = hist > 0
    chi2 = float(
        np.sum(y[mask] / hist[mask] - hist[mask]) + (z / zero - zero if zero > 0 else 0)
    )
    return chi2, ndf


def modified_neyman_chi2_B(hist, y, zero, z, dof):
    """Modified Neyman chi-square B:  sum((y-n)^2 / n)."""
    ndf = len(hist) - dof
    mask = hist > 0
    chi2 = float(
        np.sum((y[mask] - hist[mask]) ** 2 / hist[mask])
        + ((z - zero) ** 2 / zero if zero > 0 else 0)
    )
    return chi2, ndf


def mighell_chi2(hist, y, zero, z, dof):
    """Mighell chi-square:  sum((y+1-n)^2 / (n+1))."""
    ndf = len(hist) - dof
    chi2 = float(
        np.sum((y + 1 - hist) ** 2 / (hist + 1))
        + ((z + 1 - zero) ** 2 / (zero + 1) if zero >= 0 else 0)
    )
    return chi2, ndf
