# ===========================================================================
# models/tweedie.py
#
# Compound-Poisson-Gamma fitter with Gaussian pedestal.
#
# Physical model:
#   G(q)  = sum_k  p_k * h_k(q),   h_k = g0 * f^{*k}
#   G~(w) = g0~(w) * exp( lam * (f~(w) - 1) )
#
# Pedestal (0PE response):
#   g0(q)  = N(ped_mean, ped_sigma^2)
#   g0~(w) = exp(i*w*m0 - 0.5*s0^2*w^2)
#
# SPE (Gamma):
#   f~(w) = (1 + i*theta*w)^{-alpha},
#   alpha = (spe_mean / spe_sigma)^2,   theta = spe_mean / alpha
#
# Count distribution:
#   N ~ Poisson(lam),   lam = light intensity = -log(1 - occ)
#   => G~(w) = g0~(w) * exp( lam * (f~(w) - 1) )
#
# Parameter vector:
#   [logA | ped_mean, ped_sigma | spe_mean, spe_sigma | lam]
#
# Extra-parameter registry:
#   PEDESTAL_PARAMS  (_start_idx = 2)  — current stage
#   Append THRESHOLD_PARAMS            — future stage, no other change needed
# ===========================================================================

import numpy as np

from scipy.fft import fft
from scipy.stats import norm

from ..core.base import PMT_Fitter
from ..core.utils import compute_init, ExtraParam, PEDESTAL_PARAMS, THRESHOLD_PARAMS


# =============================================
#     Compound-Poisson-Gamma fitter
# =============================================


class Tweedie_Fitter(PMT_Fitter):
    """Compound-Poisson-Gamma PMT fitter with Gaussian pedestal.

    Parameters
    ----------
    hist : array-like
        Bin counts.  Must NOT include events outside the histogram window.
    bins : array-like
        Bin edges, length len(hist)+1.
    A : int or None
        Total events (including those outside the window).
    extra_params : list[ExtraParam]
        Extra parameter registry.  Defaults to PEDESTAL_PARAMS.
        Append THRESHOLD_PARAMS when that effect is ready.
    spe_init : (float, float)
        Initial (spe_mean, spe_sigma).
    spe_bounds : ((lo, hi), (lo, hi))
        Bounds for (spe_mean, spe_sigma).
    lam_init : float or None
        Initial light intensity lam = -log(1 - occ).
    q_min : float or None
        Left edge of the FFT grid.  Pass charges.min() so xsp covers the
        pedestal region when ped_mean < bins[0].  Defaults to bins[0].
    sample : int or None
        Sub-sampling factor per bin.
    seterr : str
        numpy error mode.
    fit_total : bool
        Fit logA as a free parameter.
    auto_init : bool
        Estimate pedestal and SPE parameters from a 4-Gaussian mixture fit
        before the main optimisation.
    constraints : list of dict or None
        Linear constraints on [spe_params..., lam].
    """

    _DEFAULT_SPE_INIT = (6000, 500)
    _DEFAULT_SPE_BOUNDS = ((1e-6, 15000), (1e-6, None))
    _DEFAULT_CONSTRAINTS = [
        {"coeffs": [(1, 1), (2, -1)], "threshold": 0, "op": ">"},
    ]

    def __init__(
        self,
        hist,
        bins,
        A=None,
        extra_params: list[ExtraParam] = None,
        spe_init=None,
        spe_bounds=None,
        lam_init=None,
        q_min=None,
        sample=None,
        seterr: str = "warn",
        fit_total: bool = True,
        auto_init: bool = False,
        use_threshold: bool = False,
        constraints=None,
    ):
        if extra_params is None:
            extra_params = list(PEDESTAL_PARAMS)
        if use_threshold:
            extra_params = list(extra_params) + THRESHOLD_PARAMS
        self._extra_params = extra_params
        self._use_threshold = use_threshold
        self._n_ped = len(PEDESTAL_PARAMS)

        spe_init = spe_init or self._DEFAULT_SPE_INIT
        spe_bounds = spe_bounds or self._DEFAULT_SPE_BOUNDS

        super().__init__(
            hist=hist,
            bins=bins,
            A=A,
            lam_init=lam_init,
            sample=sample,
            q_min=q_min,
            init=[ep.init for ep in extra_params] + list(spe_init),
            bounds=[ep.bound for ep in extra_params] + list(spe_bounds),
            constraints=constraints or self._DEFAULT_CONSTRAINTS,
            seterr=seterr,
            fit_total=fit_total,
        )

        self._start_idx = len(extra_params)

        if auto_init:
            self._run_auto_init()

        self._finalize_init()

    # ==============================
    #     Pedestal pipeline
    # ==============================

    def _make_b_sp(self):
        """FFT of the Gaussian pedestal g0."""

        def b_sp(args):
            extra = args[self._extra_slice()]
            pdf = self._pdf_extra(extra)
            pdf_padded = np.pad(pdf, (0, self._pad_safe), mode="constant")
            padded = np.roll(pdf_padded, -self._i_zero)
            return fft(padded) * self._xsp_width

        return b_sp

    def _make_all_PE_processor(self):
        """G~(w) = g0~(w) * exp( lam * (f~(w) - 1) ).

        In the FFT representation:
            b_sp = FFT(g0),   s_sp = f~(w)
            result = exp( lam * (s_sp - 1) ) * b_sp
        """
        return lambda lam, b_sp: lambda s_sp: np.exp(lam * (s_sp - 1)) * b_sp

    # ==============================
    #     SPE characteristic function
    # ==============================

    def _ser_ft(self, freq, ser_args):
        """Analytic Gamma CF:  f~(w) = (1 + i*theta*w)^{-alpha}."""
        spe_mean, spe_sigma = ser_args
        alpha = (spe_mean / spe_sigma) ** 2
        theta = spe_mean / alpha
        return (1.0 + 1j * theta * freq) ** (-alpha)

    # ==============================
    #     Pedestal PDF
    # ==============================

    def _pdf_extra(self, extra_args):
        return norm.pdf(self.xsp, loc=float(extra_args[0]), scale=float(extra_args[1]))

    # ==============================
    #     Gain
    # ==============================

    def get_gain(self, ser_args, gain: str = "gm"):
        spe_mean, spe_sigma = ser_args
        alpha = (spe_mean / spe_sigma) ** 2
        theta = spe_mean / alpha
        if gain == "gm":
            return float(spe_mean)
        elif gain == "gp":
            return float((alpha - 1) * theta)
        raise ValueError(f"Unknown gain type: {gain!r}")

    # ==============================
    #     Auto-initialisation
    # ==============================

    def _run_auto_init(self):
        ped_mean, ped_sigma, spe_mean, spe_sigma = compute_init(self.hist, self.bins)
        self._init[0] = ped_mean
        self._init[1] = ped_sigma
        self._init[2] = spe_mean
        self._init[3] = spe_sigma
        ped_mean_fluc = 400
        self._bounds_in[0] = (ped_mean - ped_mean_fluc, ped_mean + ped_mean_fluc)
        self._bounds_in[1] = (0, None)
        self._bounds_in[2] = (0.5 * spe_mean, 1.5 * spe_mean)
        self._bounds_in[3] = (0, spe_mean)

    # ==============================
    #     Reporting
    # ==============================

    def extra_param_names(self) -> list:
        return [ep.name for ep in self._extra_params]
