#!/usr/bin/env python3
import numpy as np

from math import log, exp
from scipy.fft import fft, ifft
from scipy.stats import norm

from .utils import (
    composite_simpson,
    isParamsInBound,
    isParamsWithinConstraints,
    merged_pearson_chi2,
    modified_neyman_chi2_A,
    modified_neyman_chi2_B,
    mighell_chi2,
)


class PMT_Fitter:
    """FFT-based PMT charge spectrum fitter (base class).

    The full observed charge density is::

        G(q) = sum_k  p_k * h_k(q),   h_k = g0 * f^{*k}

    where g0 is the pedestal (0PE response) and f is the SPE response.
    The pedestal is always convolved into every k-PE component, so it always
    contributes to the observable spectrum over the histogram window.

    The light intensity lam = -log(1 - occ) is the primary fit parameter
    for the count distribution.  It is unbounded above, making it suitable
    for high-occupancy regimes.

    Events outside the histogram window are counted as
        self.zero = A - sum(hist)
    and their expected number is estimated as
        z_est = A_now - sum(y_est).

    Parameter vector layout (flat, as seen by the optimiser)::

        [logA,  extra_0..extra_{S-1},  spe_0..spe_{M-1},  lam]

    S = _start_idx is set by the subclass after calling super().__init__(),
    which must then call _finalize_init() to assemble vectors and build the
    pipeline.

    Parameters
    ----------
    hist : array-like
        Bin counts.  Must NOT include events outside the window; those enter
        via self.zero = A - sum(hist).
    bins : array-like
        Bin edges, length len(hist)+1.
    A : int or None
        Total events including those outside the histogram window.
        Defaults to sum(hist) when None.
    lam_init : float or None
        Initial light intensity lam = -log(1 - occ).  Inferred from A and
        hist when None.
    sample : int or None
        Sub-sampling factor per bin for the FFT grid.
    init : array-like
        Initial values for [extra_params..., spe_params...].
        logA and lam are appended automatically inside _finalize_init().
    bounds : list of (lo, hi)
        Bounds matching init.  logA and lam bounds are appended automatically.
    constraints : list of dict or None
        Linear constraints on [spe_params..., lam].
    q_min : float or None
        Left edge of the FFT grid.  Pass charges.min() so xsp extends far
        enough left to cover the pedestal when ped_mean < bins[0].
        Defaults to bins[0] (no leftward extension) when None.
    fit_total : bool
        Whether to include logA as a free parameter.
    """

    def __init__(
        self,
        hist,
        bins,
        A=None,
        lam_init=None,
        sample=None,
        q_min=None,
        use_integration=False,
        init=None,
        bounds=None,
        constraints=None,
        seterr: str = "warn",
        fit_total: bool = True,
    ):
        np.seterr(all=seterr)
        self.seterr = seterr
        self._fit_total = fit_total
        self._use_integration = use_integration

        self.hist = np.asarray(hist, dtype=float)
        self.bins = np.asarray(bins, dtype=float)
        self.A = int(A)
        self.zero = self.A - int(self.hist.sum())

        if lam_init is not None:
            self._lam_init = float(lam_init)
        else:
            occ_est = float(self.hist.sum()) / self.A
            self._lam_init = -log(1.0 - min(occ_est, 1.0 - 1e-9))

        occ_est = 1.0 - exp(-self._lam_init)
        if not use_integration:
            # one sub-point per bin: xsp coincides with bin centres
            self.sample = 1
        elif sample is not None:
            self.sample = int(sample)
        else:
            self.sample = 16 * int(1 / (1 - occ_est) ** 0.673313)

        # subclass sets this to len(extra_params) after super().__init__(),
        # then calls _finalize_init()
        self._start_idx = 0

        self._init = np.asarray(init, dtype=float)
        self._bounds_in = list(bounds)
        self.constraints = constraints or []

        # ==============================
        #     FFT grid
        # ==============================

        self._bin_width = float(self.bins[1] - self.bins[0])
        self._xsp_width = self._bin_width / self.sample
        _q_min = float(q_min) if q_min is not None else -10000

        # _shift: number of sub-bins from xsp[0] to bins[0]
        self._shift = np.ceil((self.bins[0] - _q_min) / self._xsp_width).astype(int)

        self.xsp = np.linspace(
            self.bins[0] - self._shift * self._xsp_width,
            self.bins[-1],
            num=len(self.hist) * self.sample + abs(self._shift) + 1,
            endpoint=True,
        )

        # _i_zero: index in xsp where q=0 lives (for FFT origin alignment)
        # roll by +_i_zero before FFT so q=0 maps to index 0
        self._i_zero = int(round(-self.xsp[0] / self._xsp_width))

        # after IFFT rolls by -_i_zero, bins[0] is at this index in the result
        self._bins0_idx = self._shift - self._i_zero

        n_origin = len(self.xsp)
        self._n_full = 2 ** np.ceil(np.log2(n_origin)).astype(int)
        self._pad_safe = self._n_full - n_origin
        self._freq = 2 * np.pi * np.fft.fftfreq(self._n_full, d=self._xsp_width)
        self._shift_padded = (
            -self._i_zero
        )  # roll by +_i_zero before FFT so q=0 → index 0
        self._recover_slice = slice(0, n_origin)

        self._C = self._log_l_C()

    # =============================================
    #     Init assembly (called by subclass)
    # =============================================

    def _finalize_init(self):
        """Assemble full init/bounds vectors and build the pipeline.

        Must be called by the subclass once _start_idx is final.
        """
        lam_lo = 1e-06
        lam_hi = 10

        init_full = list(self._init) + [self._lam_init]
        bounds_full = list(self._bounds_in) + [(lam_lo, lam_hi)]

        if self._fit_total:
            init_full = [log(self.A)] + init_full
            bounds_full = [(None, None)] + bounds_full

        self.init = np.array(init_full, dtype=float)
        self.bounds = tuple(bounds_full)
        self.dof = len(self.init)

        self._build_pipeline()

        for v, (lo, hi) in zip(self.init, self.bounds):
            lo_s = f"{lo:.4g}" if lo is not None else "-inf"
            hi_s = f"{hi:.4g}" if hi is not None else "+inf"
            print(f"[INIT] {float(v):.4g}  in  [{lo_s}, {hi_s}]", flush=True)

    # ==============================
    #     Index helpers
    # ==============================

    def _head(self):
        return 1 if self._fit_total else 0

    def _extra_slice(self):
        h = self._head()
        return slice(h, h + self._start_idx)

    def _spe_slice(self):
        h = self._head()
        return slice(h + self._start_idx, -1)

    # ==============================
    #     Pipeline assembly
    # ==============================

    def _build_pipeline(self):
        """Build all pipeline closures.  Called once _start_idx is final."""
        self._b_sp = self._make_b_sp()
        self._all_PE_processor = self._make_all_PE_processor()
        self._nPE_processor = self._make_nPE_processor()
        self._ser_to_ft = self._make_ser_to_ft()
        self._ifft_pipeline = self._make_ifft_pipeline()
        self._pdf_sr_n = self._make_pdf_sr_n()
        self._estimate_count = self._make_estimate_counter()
        self._constraint_check = self._make_constraint_checker()

    # ==============================
    #     Pipeline factories
    # ==============================

    def _make_b_sp(self):
        """FFT of the pedestal PDF.  Subclass overrides when a pedestal is present."""
        return lambda args: None

    def _make_all_PE_processor(self):
        """Full G~(w) = g0~(w) * G_N(f~(w)) processor.
        Subclass must override to implement the count-distribution pgf.
        """
        raise NotImplementedError

    def _make_nPE_processor(self):
        """Returns a closure for the n-PE contribution in Fourier space."""
        return (
            lambda lam, n: lambda s_sp: exp(-lam)
            * (lam * s_sp) ** n
            / np.prod(range(1, n + 1))
        )

    def _make_ser_to_ft(self):
        def ser_to_ft(ser_args):
            ft = self._ser_ft(self._freq, ser_args)
            if ft is not None:
                return ft
            pdf = self._ser_pdf_time(ser_args)
            pdf_padded = np.pad(pdf, (0, self._pad_safe), mode="constant")
            padded = np.roll(pdf_padded, -self._i_zero)
            return fft(padded) * self._xsp_width

        return ser_to_ft

    def _make_ifft_pipeline(self):
        def ifft_back(s_processed):
            result = np.real(ifft(s_processed)) / self._xsp_width
            return np.maximum(np.roll(result, self._i_zero)[self._recover_slice], 0.0)

        return ifft_back

    def _make_pdf_sr_n(self):
        """Returns a closure for the n-PE component of G(q) on xsp."""
        n_spe = len(self._init) - self._start_idx

        def pdf_sr_n(args, n):
            a = np.asarray(args, float)
            if a.size == len(self.init):
                # full vector: [logA, extra..., spe..., lam]
                ser_args = a[self._spe_slice()]
                lam = float(a[-1])
                extra = a[self._extra_slice()]
            elif a.size == n_spe + 1:
                # short vector: [spe..., lam]
                ser_args = a[:-1]
                lam = float(a[-1])
                extra = np.array([])
            else:
                raise ValueError(
                    f"args length {a.size} invalid; "
                    f"expected full ({len(self.init)}) or tail ({n_spe + 1})."
                )
            if n == 0:
                return (
                    self._pdf_extra(extra)
                    if self._start_idx > 0 and extra.size > 0
                    else np.zeros_like(self.xsp)
                )
            b_sp = self._b_sp(a)
            ft = self._ser_to_ft(ser_args) + self.const(ser_args)
            return self._ifft_pipeline(self._nPE_processor(lam, n)(ft) * b_sp)

        return pdf_sr_n

    def _make_estimate_counter(self):
        """Returns a closure estimating bin counts y_est and out-of-window z_est.

        When use_integration=False, counts are estimated as pdf * bin_width
        evaluated at bin centres (xsp has sample=1, one point per bin).
        When use_integration=True, composite Simpson's rule is used.
        """
        need_mask = self.bins[0] == 0

        if self._use_integration:
            # composite-Simpson weights for one bin (sample+1 evaluation points)
            w = np.ones(self.sample + 1)
            w[1:-1:2] = 4
            w[2:-2:2] = 2
            w *= self._xsp_width / 3
            self._simp_w = w

            def counter(args):
                A_now = self._A_from_args(args)
                y_sp = A_now * self._pdf_sr(args)
                if need_mask:
                    y_sp[0] = 0.0
                nbin = len(self.hist)
                start = self._bins0_idx
                idx = (
                    start
                    + self.sample * np.arange(nbin)[:, None]
                    + np.arange(self.sample + 1)[None, :]
                )
                y_est = np.maximum(y_sp[idx] @ self._simp_w, 1e-32)
                z_est = max(A_now - float(y_est.sum()), 1e-32)
                return y_est, z_est

        else:
            # pdf * bin_width at bin centres (sample=1, xsp[_bins0_idx + i] ≈ centre of bin i)
            def counter(args):
                A_now = self._A_from_args(args)
                y_sp = A_now * self._pdf_sr(args)
                if need_mask:
                    y_sp[0] = 0.0
                nbin = len(self.hist)
                start = self._bins0_idx
                # with sample=1, xsp[start + i] is the left edge of bin i;
                # use midpoint: offset by half a sub-bin
                idx = start + np.arange(nbin)
                y_est = np.maximum(y_sp[idx] * self._bin_width, 1e-32)
                z_est = max(A_now - float(y_est.sum()), 1e-32)
                return y_est, z_est

        return counter

    def _make_constraint_checker(self):
        if not self.constraints:
            return lambda args: True
        return lambda args: isParamsWithinConstraints(
            args[self._head() + self._start_idx :], self.constraints
        )

    # ==============================
    #     PDF evaluation
    # ==============================

    def _pdf_sr(self, args):
        """Full spectrum PDF G(q) evaluated on xsp."""
        ser_args = args[self._spe_slice()]
        lam = float(args[-1])
        b_sp = self._b_sp(args)
        ft = self._ser_to_ft(ser_args) + self.const(ser_args)
        return self._ifft_pipeline(self._all_PE_processor(lam, b_sp)(ft))

    def _pdf_extra(self, extra_args):
        """Pedestal (0PE) PDF on xsp.  Subclass overrides when a pedestal is present."""
        return np.zeros_like(self.xsp)

    def _estimate_smooth(self, args):
        return self._A_from_args(args) * self._bin_width * self._pdf_sr(args)

    def estimate_smooth_n(self, n):
        """Expected count density for exactly n PE, using fitted full_args."""
        return (
            self._A_from_args(self.full_args)
            * self._bin_width
            * self._pdf_sr_n(self.full_args, n)
        )

    def estimate_count_n(self, n):
        """Expected bin counts for exactly n PE (same length as hist)."""
        A_now = self._A_from_args(self.full_args)
        y_sp = A_now * self._pdf_sr_n(self.full_args, n)
        nbin = len(self.hist)
        start = self._bins0_idx

        if self._use_integration:
            idx = (
                start
                + self.sample * np.arange(nbin)[:, None]
                + np.arange(self.sample + 1)[None, :]
            )
            return np.maximum(y_sp[idx] @ self._simp_w, 0.0)
        else:
            idx = start + np.arange(nbin)
            return np.maximum(y_sp[idx] * self._bin_width, 0.0)

    # ==============================
    #     Likelihood
    # ==============================

    def _log_l_C(self):
        """Factorial constant in the extended Poisson log-likelihood."""
        n_part = sum(float(np.sum(np.log(np.arange(1, int(n) + 1)))) for n in self.hist)
        n0_part = float(np.sum(np.log(np.arange(1, int(self.zero) + 1))))
        return n_part + n0_part

    def log_l(self, args) -> float:
        if not (isParamsInBound(args, self.bounds) and self._constraint_check(args)):
            return -np.inf
        y, z = self._estimate_count(args)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = np.sum(self.hist * np.log(y) - y) + self.zero * np.log(z) - z - self._C
        return float(ll) if np.isfinite(ll) else -np.inf

    def get_chi_sq(self, args, chi_sq_fn, dof):
        y, z = self._estimate_count(args)
        return chi_sq_fn(self.hist, y, self.zero, z, dof)

    # ==============================
    #     Fitting
    # ==============================

    def fit(self, *, strategy=1, tol=1e-1, max_calls=10000, print_level=0):
        """Fit with Minuit (pyROOT backend).

        Parameters
        ----------
        strategy : int
            Minuit strategy (0, 1, or 2).
        tol : float
            Convergence tolerance.
        max_calls : int
            Maximum number of function calls.
        print_level : int
            Minuit verbosity.
        """
        import ROOT

        ROOT.gErrorIgnoreLevel = ROOT.kError

        def _nll(par_ptr):
            args = np.array([par_ptr[i] for i in range(self.dof)], dtype=float)
            ll = self.log_l(args)
            return 1e30 if not np.isfinite(ll) else -ll

        self._nll = _nll  # keep reference alive for ROOT GC
        fcn = ROOT.Math.Functor(self._nll, self.dof)

        def _setup(m, this_strat, this_tol):
            m.SetFunction(fcn)
            m.SetStrategy(this_strat)
            m.SetErrorDef(0.5)
            m.SetTolerance(this_tol)
            m.SetMaxFunctionCalls(max_calls)
            m.SetPrintLevel(print_level)
            for i, (v0, (lo, hi)) in enumerate(zip(self.init, self.bounds)):
                step = 0.1 * (abs(float(v0)) or 1.0)
                if lo is None and hi is None:
                    m.SetVariable(i, f"p{i}", float(v0), step)
                elif lo is not None and hi is not None:
                    m.SetLimitedVariable(i, f"p{i}", float(v0), step, lo, hi)
                elif lo is not None:
                    m.SetLowerLimitedVariable(i, f"p{i}", float(v0), step, lo)
                else:
                    m.SetUpperLimitedVariable(i, f"p{i}", float(v0), step, hi)

        algos = ["Migrad", "Combined", "Migrad", "Combined", "Migrad", "Combined"]
        for attempt, algo in enumerate(algos):
            this_tol = tol if attempt < 2 else min(10 * tol, 1.0)
            this_strat = (3 - strategy) if attempt >= 4 else strategy
            m = ROOT.Math.Factory.CreateMinimizer("Minuit2", algo)
            _setup(m, this_strat, this_tol)
            if m.Minimize():
                break
            print(f"[WARN] Minuit attempt {attempt + 1} failed", flush=True)
        else:
            print("[WARN] Minuit did not converge after 6 attempts", flush=True)

        m.Hesse()
        self.converged = m.Status() == 0

        full = np.array([m.X()[i] for i in range(self.dof)])
        full_e = np.array([m.Errors()[i] for i in range(self.dof)])
        self._store_results(full, full_e, -m.MinValue())

    # ==============================
    #     Result storage
    # ==============================

    def _store_results(self, full, full_e, likelihood):
        h, s = self._head(), self._start_idx
        self.full_args = full
        self.full_args_std = full_e
        self.extra_args = full[h : h + s]
        self.extra_args_std = full_e[h : h + s]
        self.ser_args = full[h + s : -1]
        self.ser_args_std = full_e[h + s : -1]
        self.lam = float(full[-1])
        self.lam_std = float(full_e[-1])
        self.gps = self.get_gain(self.ser_args, "gp")
        self.gms = self.get_gain(self.ser_args, "gm")

        self.likelihood = likelihood
        self.smooth = self._estimate_smooth(full)
        self.ys, self.zs = self._estimate_count(full)

        self.chi_sq_pearson, self.ndf_merged = self.get_chi_sq(
            full, merged_pearson_chi2, self.dof
        )
        self.chi_sq_neyman_A, self.ndf = self.get_chi_sq(
            full, modified_neyman_chi2_A, self.dof
        )
        self.chi_sq_neyman_B, _ = self.get_chi_sq(
            full, modified_neyman_chi2_B, self.dof
        )
        self.chi_sq_mighell, _ = self.get_chi_sq(full, mighell_chi2, self.dof)

        self.aic = 2 * self.dof - 2 * self.likelihood
        self.bic = self.dof * np.log(self.A) - 2 * self.likelihood

        for name, v, e in zip(
            self.extra_param_names(), self.extra_args, self.extra_args_std
        ):
            print(f"[RES] {name}: {v:.4g} ± {e:.4g}", flush=True)
        for i, (v, e) in enumerate(zip(self.ser_args, self.ser_args_std)):
            print(f"[RES] spe[{i}]: {v:.4g} ± {e:.4g}", flush=True)
        print(f"[RES] lam: {self.lam:.4g} ± {self.lam_std:.4g}", flush=True)

    # ==============================
    #     Subclass interface
    # ==============================

    def _ser_ft(self, freq, ser_args):
        """Analytic SPE characteristic function.  Return None to use FFT fallback."""
        return None

    def _ser_pdf_time(self, ser_args):
        raise NotImplementedError

    def const(self, ser_args):
        """Delta-at-zero weight in the SPE (e.g. for models with a δ component)."""
        return 0

    def get_gain(self, ser_args, gain: str = "gm"):
        raise NotImplementedError

    def extra_param_names(self) -> list:
        """Human-readable names for extra_args, used in result reporting."""
        return [f"extra[{i}]" for i in range(self._start_idx)]

    def _replace_spe_params(self, mean_init, sigma_init):
        raise NotImplementedError

    def _replace_spe_bounds(self, mean_bound, sigma_bound):
        raise NotImplementedError

    # ==============================
    #     Utilities
    # ==============================

    def _A_from_args(self, args):
        if self._fit_total:
            return float(np.exp(np.clip(float(args[0]), 0, log(1e12))))
        return float(self.A)
