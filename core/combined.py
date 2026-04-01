# ===========================================================================
# core/combined.py
#
# Joint fitter for multiple spectra sharing one set of SER + pedestal params.
#
# Parameter vector layout:
#   theta = [logA_1, ..., logA_N,        (independent, one per spectrum)
#            extra_0, ..., extra_{S-1},   (shared pedestal, length _start_idx)
#            spe_0,  ..., spe_{M-1},      (shared SER)
#            lam_1,  ..., lam_N]          (independent, one per spectrum)
#
# All individual fitters must have identical _fit_total, _start_idx, and
# SER dimension.  Shared parameters are taken from fitters[0].
# ===========================================================================

from __future__ import annotations

import numpy as np
from typing import List


class CombinedFitter:
    """Combine multiple spectra into a joint fit with shared SER + pedestal.

    Parameters
    ----------
    fitters : list of PMT_Fitter
        Individual fitters, each already constructed with hist/bins/A/q_min.
        Must all have the same _fit_total, _start_idx, and SER dimension.
    """

    def __init__(self, fitters: List[object], ref_index: int = 0):
        if not fitters:
            raise ValueError("At least one fitter is required.")
        if not (0 <= ref_index < len(fitters)):
            raise ValueError(
                f"ref_index={ref_index} out of range for {len(fitters)} fitters."
            )

        self.fitters = fitters
        self.n_spectra = len(fitters)
        self._ref_index = ref_index

        self._validate_fitters()
        self._build_parameter_structure()

        print(f"\n[INFO] Combined {self.n_spectra} spectra", flush=True)
        print(f"[INFO] Total parameters: {self.dof}", flush=True)

    # ==============================
    #     Validation
    # ==============================

    def _validate_fitters(self):
        ref = self.fitters[self._ref_index]
        ref_head = 1 if ref._fit_total else 0
        ref_ser_dim = len(ref.init) - ref_head - ref._start_idx - 1

        for i, f in enumerate(self.fitters):
            if i == self._ref_index:
                continue
            if f._fit_total != ref._fit_total:
                raise ValueError(
                    f"Fitter {i}: fit_total={f._fit_total}, expected {ref._fit_total}"
                )
            if f._start_idx != ref._start_idx:
                raise ValueError(
                    f"Fitter {i}: _start_idx={f._start_idx}, expected {ref._start_idx}"
                )
            head = 1 if f._fit_total else 0
            ser_dim = len(f.init) - head - f._start_idx - 1
            if ser_dim != ref_ser_dim:
                raise ValueError(
                    f"Fitter {i}: SER dimension={ser_dim}, expected {ref_ser_dim}"
                )

    # ==============================
    #     Parameter structure
    # ==============================

    def _build_parameter_structure(self):
        ref = self.fitters[self._ref_index]

        self._fit_total = ref._fit_total
        self._start_idx = ref._start_idx
        self._head = 1 if self._fit_total else 0

        init_parts = []
        bounds_parts = []
        cursor = 0

        # --- logA: independent per spectrum ---
        if self._fit_total:
            self._logA_indices = []
            for f in self.fitters:
                init_parts.append(np.array([f.init[0]]))
                bounds_parts.append(f.bounds[0])
                self._logA_indices.append(cursor)
                cursor += 1
        else:
            self._logA_indices = []

        # --- shared extra (pedestal) params ---
        if self._start_idx > 0:
            extra_slice = slice(self._head, self._head + self._start_idx)
            init_parts.append(ref.init[extra_slice])
            bounds_parts.extend(ref.bounds[extra_slice])
            self._extra_slice = slice(cursor, cursor + self._start_idx)
            cursor += self._start_idx
        else:
            self._extra_slice = slice(0, 0)

        # --- shared SER params ---
        ser_start = self._head + self._start_idx
        ser_end = len(ref.init) - 1  # last element is lam
        self._ser_len = ser_end - ser_start
        init_parts.append(ref.init[ser_start:ser_end])
        bounds_parts.extend(ref.bounds[ser_start:ser_end])
        self._ser_slice = slice(cursor, cursor + self._ser_len)
        cursor += self._ser_len

        # --- lam: independent per spectrum ---
        self._lam_indices = []
        for f in self.fitters:
            init_parts.append(np.array([f.init[-1]]))
            bounds_parts.append(f.bounds[-1])
            self._lam_indices.append(cursor)
            cursor += 1

        self.init = np.concatenate(init_parts)
        self.bounds = tuple(bounds_parts)
        self.dof = len(self.init)

        # Remap constraints from ref fitter local SER indices → combined indices
        self.constraints = self._remap_constraints(ref.constraints)

    def _remap_constraints(self, constraints):
        """Remap constraint indices from local SER space to combined space."""
        if not constraints:
            return []

        local_ser_start = self._head + self._start_idx
        combined_ser_start = self._ser_slice.start
        offset = combined_ser_start - local_ser_start

        remapped = []
        for c in constraints:
            if isinstance(c, dict):
                nc = c.copy()
                nc["coeffs"] = [(idx + offset, coeff) for idx, coeff in c["coeffs"]]
                remapped.append(nc)
            else:
                raise ValueError(f"Unknown constraint format: {c}")
        return remapped

    # ==============================
    #     Local arg builder
    # ==============================

    def _build_local_args(self, theta: np.ndarray, i: int) -> np.ndarray:
        """Reconstruct the individual fitter's full parameter vector from theta."""
        parts = []

        # logA (individual)
        if self._fit_total:
            parts.append(theta[self._logA_indices[i] : self._logA_indices[i] + 1])

        # shared extra (pedestal)
        if self._start_idx > 0:
            parts.append(theta[self._extra_slice])

        # shared SER
        parts.append(theta[self._ser_slice])

        # lam (individual)
        parts.append(theta[self._lam_indices[i] : self._lam_indices[i] + 1])

        return np.concatenate(parts)

    # ==============================
    #     Likelihood
    # ==============================

    def log_l(self, theta: np.ndarray) -> float:
        """Joint log-likelihood = sum of individual log-likelihoods."""
        total = 0.0
        for i, f in enumerate(self.fitters):
            ll = f.log_l(self._build_local_args(theta, i))
            if not np.isfinite(ll):
                return -np.inf
            total += ll
        return total

    # ==============================
    #     Fitting
    # ==============================

    def fit(self, strategy=1, tol=1e-1, max_calls=10000, print_level=0):
        """Fit with Minuit (pyROOT backend)."""
        import ROOT

        ROOT.gErrorIgnoreLevel = ROOT.kError

        def _nll(par_ptr):
            args = np.array([par_ptr[i] for i in range(self.dof)], dtype=float)
            ll = self.log_l(args)
            return 1e30 if not np.isfinite(ll) else -ll

        self._nll = _nll
        fcn = ROOT.Math.Functor(self._nll, self.dof)

        def _setup(m, strat, this_tol):
            m.SetFunction(fcn)
            m.SetStrategy(strat)
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

        fitted = np.array([m.X()[i] for i in range(self.dof)])
        errors = np.array([m.Errors()[i] for i in range(self.dof)])

        self._store_results(fitted, errors, -m.MinValue())
        self._minimizer = m

    # ==============================
    #     Result storage
    # ==============================

    def _store_results(self, fitted, errors, likelihood):
        self.fitted_params = fitted
        self.param_errors = errors
        self.likelihood = likelihood

        # shared extra (pedestal)
        self.extra_args = fitted[self._extra_slice]
        self.extra_args_std = errors[self._extra_slice]

        # shared SER
        self.ser_args = fitted[self._ser_slice]
        self.ser_args_std = errors[self._ser_slice]

        # per-spectrum lam
        self.lams = fitted[self._lam_indices]
        self.lams_std = errors[self._lam_indices]

        # per-spectrum logA (if fit_total)
        if self._fit_total:
            self.logAs = fitted[self._logA_indices]
            self.logAs_std = errors[self._logA_indices]

        # info criteria
        n_data = sum(len(f.hist) + 1 for f in self.fitters)
        self.aic = 2 * self.dof - 2 * self.likelihood
        self.bic = self.dof * np.log(n_data) - 2 * self.likelihood

        # correlation of shared SER block
        s = self._ser_slice
        self.spe_corr = (
            np.array(
                [
                    [self._minimizer.Correlation(i, j) for j in range(s.start, s.stop)]
                    for i in range(s.start, s.stop)
                ]
            )
            if hasattr(self, "_minimizer")
            else None
        )

        # gains via fitters[0]
        ref = self.fitters[self._ref_index]
        self.gps = ref.get_gain(self.ser_args, "gp")
        self.gms = ref.get_gain(self.ser_args, "gm")

        self._print_results()

    def _print_results(self):
        print(f"\n[INFO] Converged : {self.converged}", flush=True)
        print(f"[INFO] Log-L     : {self.likelihood:.4f}", flush=True)
        print(f"[INFO] AIC       : {self.aic:.4f}", flush=True)
        print(f"[INFO] BIC       : {self.bic:.4f}", flush=True)

        def _fmt(name, v, e, lo, hi):
            at = ""
            if lo is not None and abs(v - lo) < 1e-6 * (abs(lo) or 1):
                at = "  [AT LOWER BOUND]"
            elif hi is not None and abs(v - hi) < 1e-6 * (abs(hi) or 1):
                at = "  [AT UPPER BOUND]"
            print(f"[INFO] {name}: {v:.4g} ± {e:.4g}{at}", flush=True)

        bounds_list = list(self.bounds)
        if self._start_idx > 0:
            names = self.fitters[0].extra_param_names()
            print("[INFO] Shared pedestal:", flush=True)
            for i, (name, v, e) in enumerate(
                zip(names, self.extra_args, self.extra_args_std)
            ):
                lo, hi = bounds_list[self._extra_slice.start + i]
                _fmt(f"  {name}", v, e, lo, hi)

        print("[INFO] Shared SER:", flush=True)
        for i, (v, e) in enumerate(zip(self.ser_args, self.ser_args_std)):
            lo, hi = bounds_list[self._ser_slice.start + i]
            _fmt(f"  spe[{i}]", v, e, lo, hi)

        print("[INFO] Per-spectrum lam:", flush=True)
        for i, (v, e) in enumerate(zip(self.lams, self.lams_std)):
            lo, hi = bounds_list[self._lam_indices[i]]
            _fmt(f"  spectrum {i}", v, e, lo, hi)
