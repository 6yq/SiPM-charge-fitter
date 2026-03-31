# ===========================================================================
# models/generalized_tweedie.py
#
# Compound-Generalized-Poisson-Gamma fitter with Gaussian pedestal.
#
# Physical model:
#   G(q)  = sum_k  p_k * h_k(q),   h_k = g0 * f^{*k}
#   G~(w) = g0~(w) * exp( lam * (T(f~(w)) - 1) )
#
# Pedestal and SPE are identical to tweedie.py.
#
# Count distribution — Generalized Poisson(lam, xi):
#   pgf:  G_N(s) = exp( lam * (T(s) - 1) )
#   where T(s) solves  T = s * exp(xi*(T-1))
#   Lambert W form:   T(s) = -W(-xi * s * e^{-xi}) / xi
#
# Parameter vector:
#   [logA | ped_mean, ped_sigma | spe_mean, spe_sigma, xi | lam]
#
# xi = 0 recovers the plain Poisson case.
# ===========================================================================

import numpy as np

from scipy.fft import fft
from scipy.special import lambertw
from scipy.stats import norm

from ..core.base import PMT_Fitter
from ..core.utils import compute_init, ExtraParam, PEDESTAL_PARAMS


# =================================================
#     Compound-Generalized-Poisson-Gamma fitter
# =================================================


class Gen_Tweedie_Fitter(PMT_Fitter):
    """Compound-Generalized-Poisson-Gamma PMT fitter with Gaussian pedestal.

    Identical to Tweedie_Fitter except that the photon-count distribution is
    Generalized Poisson(lam, xi) rather than plain Poisson(lam).  The
    dispersion parameter xi is the third SPE-block parameter.

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
    spe_init : (float, float, float)
        Initial (spe_mean, spe_sigma, xi).
    spe_bounds : ((lo,hi), (lo,hi), (lo,hi))
        Bounds for (spe_mean, spe_sigma, xi).
    lam_init : float or None
        Initial light intensity.
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
        Estimate pedestal and SPE parameters from a 4-Gaussian mixture fit.
        xi is left at its default initial value.
    constraints : list of dict or None
        Linear constraints on [spe_params..., lam].
    """

    _DEFAULT_SPE_INIT = (1.0, 0.3, 0.03)
    _DEFAULT_SPE_BOUNDS = ((1e-6, None), (1e-6, None), (1e-6, 0.999))
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
        constraints=None,
    ):
        if extra_params is None:
            extra_params = PEDESTAL_PARAMS
        self._extra_params = extra_params

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
        """G~(w) = g0~(w) * exp( lam * (T(f~(w)) - 1) ).

        In the FFT representation:
            b_sp = FFT(g0),   s_sp = f~(w)
            result = exp( lam * (T(s_sp) - 1) ) * b_sp
        """

        def processor(lam, b_sp):
            def apply(s_sp):
                return np.exp(lam * (self._T(s_sp) - 1.0)) * b_sp

            return apply

        return processor

    # ==============================
    #     SPE characteristic function
    # ==============================

    def _ser_ft(self, freq, ser_args):
        """Analytic Gamma CF:  f~(w) = (1 + i*theta*w)^{-alpha}.
        Also caches xi from ser_args for use in _T().
        """
        spe_mean, spe_sigma, xi = ser_args
        self._xi = float(xi)
        alpha = (spe_mean / spe_sigma) ** 2
        theta = spe_mean / alpha
        return (1.0 + 1j * theta * freq) ** (-alpha)

    def _T(self, s):
        """Solve T = s * exp(xi*(T-1)) via the principal branch of Lambert W.

        T(s) = -W(-xi * s * e^{-xi}) / xi

        Note: s is complex (Fourier domain), so T is complex too.
        The .real part is taken only after the final IFFT, not here.
        """
        xi = self._xi
        return -lambertw(-xi * s * np.exp(-xi), k=0) / xi

    def _make_nPE_processor(self):
        """n-PE contribution using Generalized Poisson PMF.

        p_k = lam * (lam + xi*k)^{k-1} * exp(-lam - xi*k) / k!   for k >= 1
        p_0 = exp(-lam)
        """
        from math import exp as _exp, factorial as _fac, log as _log

        def processor(lam, n):
            if n == 0:
                p_n = _exp(-lam)
            else:
                xi = self._xi
                # log p_n for numerical stability
                log_p = (
                    _log(lam)
                    + (n - 1) * _log(max(lam + xi * n, 1e-300))
                    - lam
                    - xi * n
                    - sum(_log(k) for k in range(1, n + 1))
                )
                p_n = _exp(log_p)
            return lambda s_sp: p_n * s_sp**n

        return processor

    def _pdf_extra(self, extra_args):
        return norm.pdf(self.xsp, loc=float(extra_args[0]), scale=float(extra_args[1]))

    # ==============================
    #     Gain
    # ==============================

    def get_gain(self, ser_args, gain: str = "gm"):
        spe_mean, spe_sigma, _ = ser_args
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
        self._bounds_in[1] = (0, abs(ped_mean) + ped_mean_fluc)
        self._bounds_in[2] = (0.5 * spe_mean, 1.5 * spe_mean)
        self._bounds_in[3] = (0.01 * spe_mean, spe_mean)

    # ==============================
    #     Reporting
    # ==============================

    def extra_param_names(self) -> list:
        return [ep.name for ep in self._extra_params]
