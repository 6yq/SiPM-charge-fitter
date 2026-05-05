# ===========================================================================
# core/combined.py
#
# Joint fitter for multiple SpectrumFitter spectra sharing one set of
# pedestal (extra) and SER (spe) parameters.
#
# Combined parameter vector layout:
#   theta = [log_A_0, ..., log_A_{N-1},   <- per-spectrum (N scalars)
#            extra_0, ..., extra_{S-1},    <- shared pedestal (S scalars)
#            spe_0,   ..., spe_{M-1},      <- shared SER     (M scalars)
#            lam_0,   ..., lam_{N-1}]      <- per-spectrum   (N scalars)
#
# All fitters must share the same extra_block and spe_block dimensions.
# Shared initial values are taken from fitters[0].
# ===========================================================================

from __future__ import annotations

import jax
import numpy as np
import jax.numpy as jnp

from typing import List
from scipy.optimize import minimize
from dataclasses import dataclass, field

from .base import FitResult, SpectrumFitter

jax.config.update("jax_enable_x64", True)


# ==============================
#     Combined fit result
# ==============================


@dataclass
class CombinedFitResult:
    converged: bool
    theta: np.ndarray  # combined vector, length 2N + S + M
    theta_err: np.ndarray
    logl: float
    n_iter: int
    message: str
    # slices / indices into theta for downstream use
    logA_sl: List[int]  # indices of log_A_i
    extra_sl: slice  # shared extra
    spe_sl: slice  # shared SER
    lam_sl: List[int]  # indices of lam_i

    def log_As(self):
        return self.theta[self.logA_sl], self.theta_err[self.logA_sl]

    def extra(self):
        return self.theta[self.extra_sl], self.theta_err[self.extra_sl]

    def spe(self):
        return self.theta[self.spe_sl], self.theta_err[self.spe_sl]

    def lams(self):
        return self.theta[self.lam_sl], self.theta_err[self.lam_sl]

    def local_theta(self, i, layout):
        """Reconstruct the i-th fitter's individual theta from combined theta."""
        return np.concatenate(
            [
                self.theta[self.logA_sl[i] : self.logA_sl[i] + 1],
                self.theta[self.extra_sl],
                self.theta[self.spe_sl],
                self.theta[self.lam_sl[i] : self.lam_sl[i] + 1],
            ]
        )


# ==============================
#     CombinedFitter
# ==============================


class CombinedFitter:
    """Joint MLE for multiple spectra with shared extra + SER parameters.

    Parameters
    ----------
    fitters : list of SpectrumFitter
        Each fitter must be constructed (with Q_raw, grid, likelihood closure)
        before being passed here.  All fitters must have the same extra_block
        and spe_block dimensions; shared init values are taken from fitters[0].
    """

    def __init__(self, fitters: List[SpectrumFitter]):
        if not fitters:
            raise ValueError("At least one fitter required.")
        self.fitters = fitters
        self.n_spectra = len(fitters)
        self._validate()
        self._build()

    # ==============================
    #     Validation
    # ==============================

    def _validate(self):
        ref = self.fitters[0]
        n_extra = len(ref.extra_block.init)
        n_spe = len(ref.spe_block.init)
        for i, f in enumerate(self.fitters[1:], 1):
            if len(f.extra_block.init) != n_extra:
                raise ValueError(
                    f"Fitter {i}: extra_block dim {len(f.extra_block.init)}, "
                    f"expected {n_extra}."
                )
            if len(f.spe_block.init) != n_spe:
                raise ValueError(
                    f"Fitter {i}: spe_block dim {len(f.spe_block.init)}, "
                    f"expected {n_spe}."
                )

    # ==============================
    #     Parameter structure
    # ==============================

    def _build(self):
        ref = self.fitters[0]
        n = self.n_spectra
        n_extra = len(ref.extra_block.init)
        n_spe = len(ref.spe_block.init)

        # cursor in combined theta
        # [log_A_0..N-1 | extra | spe | lam_0..N-1]
        self.logA_sl = list(range(n))
        self.extra_sl = slice(n, n + n_extra)
        self.spe_sl = slice(n + n_extra, n + n_extra + n_spe)
        self.lam_sl = list(range(n + n_extra + n_spe, 2 * n + n_extra + n_spe))

        ly = ref.layout  # {"log_A": slice, "extra": slice, "spe": slice, "lam": slice}

        init_parts = []
        bounds_parts = []

        for f in self.fitters:
            init_parts.append(f.init[ly["log_A"]])
            bounds_parts.extend(f.bounds[ly["log_A"].start : ly["log_A"].stop])

        init_parts.append(ref.init[ly["extra"]])
        bounds_parts.extend(ref.bounds[ly["extra"].start : ly["extra"].stop])

        init_parts.append(ref.init[ly["spe"]])
        bounds_parts.extend(ref.bounds[ly["spe"].start : ly["spe"].stop])

        for f in self.fitters:
            init_parts.append(f.init[ly["lam"]])
            bounds_parts.extend(f.bounds[ly["lam"].start : ly["lam"].stop])

        self.init = np.concatenate(init_parts)
        self.bounds = bounds_parts
        self.dof = len(self.init)
        self._layout_ref = ly  # single-fitter layout, kept for local_theta

    # ==============================
    #     Local theta reconstruction
    # ==============================

    def local_theta(self, theta: np.ndarray, i: int) -> np.ndarray:
        """Return individual fitter i's theta from combined theta (numpy)."""
        return np.concatenate(
            [
                theta[self.logA_sl[i] : self.logA_sl[i] + 1],
                theta[self.extra_sl],
                theta[self.spe_sl],
                theta[self.lam_sl[i] : self.lam_sl[i] + 1],
            ]
        )

    def local_theta_jax(self, theta: jnp.ndarray, i: int) -> jnp.ndarray:
        """Return individual fitter i's theta from combined theta (JAX array)."""
        return jnp.concatenate(
            [
                theta[self.logA_sl[i] : self.logA_sl[i] + 1],
                theta[self.extra_sl],
                theta[self.spe_sl],
                theta[self.lam_sl[i] : self.lam_sl[i] + 1],
            ]
        )

    # ==============================
    #     Joint log-likelihood
    # ==============================

    def _logl_combined(self, theta: jnp.ndarray) -> jnp.ndarray:
        """JAX-traceable joint log-L (sum over fitters)."""
        total = jnp.zeros(())
        for i, f in enumerate(self.fitters):
            total = total + f._logl_from_theta(self.local_theta_jax(theta, i))
        return total

    # ==============================
    #     MLE
    # ==============================

    def fit_mle(
        self, theta0: np.ndarray = None, maxiter: int = 1000
    ) -> CombinedFitResult:
        """Joint L-BFGS-B MLE with shared extra + SER.

        The gradient is assembled analytically from each fitter's individual
        gradient, avoiding a full combined Hessian at optimisation time.

        Parameters
        ----------
        theta0  : initial combined parameter vector (default: self.init)
        maxiter : L-BFGS-B iteration cap
        """
        if theta0 is None:
            theta0 = self.init.copy()
        theta0 = np.asarray(theta0, dtype=np.float64)

        # JIT warm-up so the first optimiser call is not penalised
        _th0_jax = jnp.asarray(theta0)
        for i, f in enumerate(self.fitters):
            f._logl_jit(self.local_theta_jax(_th0_jax, i))
            f._grad_jit(self.local_theta_jax(_th0_jax, i))

        logl_init = float(self._logl_combined(_th0_jax))

        # Gradient assembled per-fitter to avoid retracing the full combined fn
        def neg_logl(x: np.ndarray) -> float:
            xj = jnp.asarray(x)
            total = 0.0
            for i, f in enumerate(self.fitters):
                total += float(f._logl_jit(self.local_theta_jax(xj, i)))
            return -total

        def neg_grad(x: np.ndarray) -> np.ndarray:
            xj = jnp.asarray(x)
            g = np.zeros_like(x)
            ly = self._layout_ref
            for i, f in enumerate(self.fitters):
                lg = np.asarray(f._grad_jit(self.local_theta_jax(xj, i)))
                g[self.logA_sl[i]] += lg[ly["log_A"].start]
                g[self.extra_sl] += lg[ly["extra"]]
                g[self.spe_sl] += lg[ly["spe"]]
                g[self.lam_sl[i]] += lg[ly["lam"].start]
            return -g

        res = minimize(
            neg_logl,
            theta0,
            jac=neg_grad,
            method="L-BFGS-B",
            bounds=self.bounds,
            options={"maxiter": maxiter},
        )

        theta_hat = np.asarray(res.x, dtype=np.float64)
        logl_final = -float(res.fun)

        if logl_final < logl_init - 1.0:
            import warnings

            warnings.warn(
                f"Combined fit ended worse than init "
                f"(logl {logl_final:.2f} < init {logl_init:.2f}); "
                "result may be unreliable."
            )

        self._print_result(theta_hat, logl_init, logl_final, res)
        theta_err = self._hessian_errors(theta_hat)

        return CombinedFitResult(
            converged=bool(res.success),
            theta=theta_hat,
            theta_err=theta_err,
            logl=logl_final,
            n_iter=int(res.nit),
            message=str(res.message),
            logA_sl=self.logA_sl,
            extra_sl=self.extra_sl,
            spe_sl=self.spe_sl,
            lam_sl=self.lam_sl,
        )

    @staticmethod
    def _at_bound(v, lo, hi, rtol=1e-4):
        if lo is not None and abs(v - lo) <= rtol * (abs(lo) + 1):
            return " [AT LOWER]"
        if hi is not None and abs(v - hi) <= rtol * (abs(hi) + 1):
            return " [AT UPPER]"
        return ""

    def _print_result(self, theta, logl_init, logl_final, res):
        print(
            f"[COMB] logl  init={logl_init:.4g}  final={logl_final:.4g}"
            f"  nit={res.nit}  {res.message}",
            flush=True,
        )
        # shared extra
        for j, name in enumerate(self.fitters[0].extra_block.names):
            idx = self.extra_sl.start + j
            v = theta[idx]
            lo, hi = self.bounds[idx]
            print(
                f"  extra  {name:20s} = {v:.6g}  [{lo}, {hi}]"
                f"{self._at_bound(v, lo, hi)}",
                flush=True,
            )
        # shared spe — also print physical mean/sigma for GenTweedieFitter
        spe_theta = theta[self.spe_sl]
        spe_names = self.fitters[0].spe_block.names
        for j, name in enumerate(spe_names):
            idx = self.spe_sl.start + j
            v = theta[idx]
            lo, hi = self.bounds[idx]
            print(
                f"  spe    {name:20s} = {v:.6g}  [{lo}, {hi}]"
                f"{self._at_bound(v, lo, hi)}",
                flush=True,
            )
        # physical SPE mean/sigma if the fitter exposes spe_report
        try:
            report = self.fitters[0].spe_report(spe_theta)
            print(
                f"  spe    {'spe_mean (phys)':20s} = {report['spe_mean']:.6g}",
                flush=True,
            )
            print(
                f"  spe    {'spe_sigma (phys)':20s} = {report['spe_sigma']:.6g}",
                flush=True,
            )
        except Exception:
            pass
        # per-spectrum log_A and lam
        for i in range(self.n_spectra):
            logA = theta[self.logA_sl[i]]
            lam = theta[self.lam_sl[i]]
            lo_A, hi_A = self.bounds[self.logA_sl[i]]
            lo_lam, hi_lam = self.bounds[self.lam_sl[i]]
            print(
                f"  spec {i:2d}  log_A={logA:.6g}  [{lo_A:.4g}, {hi_A:.4g}]"
                f"{self._at_bound(logA, lo_A, hi_A)}"
                f"  lam={lam:.6g}  [{lo_lam}, {hi_lam}]"
                f"{self._at_bound(lam, lo_lam, hi_lam)}",
                flush=True,
            )

    def _hessian_errors(self, theta_hat: np.ndarray) -> np.ndarray:
        _logl_jit_combined = jax.jit(self._logl_combined)
        H = np.asarray(
            jax.hessian(_logl_jit_combined)(jnp.asarray(theta_hat, dtype=jnp.float64))
        )
        cov = -np.linalg.inv(H)
        diag = np.diag(cov)
        diag = np.where(diag > 0, diag, np.nan)
        return np.sqrt(diag)
