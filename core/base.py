# ===========================================================================
# core/base.py
#
# Bundles for one spectrum:
#   - static FFT grid,
#   - JAX-jitted extended-Poisson log-likelihood (analytic Fourier bin
#     integration, no sub-sampling),
#   - fast MLE via L-BFGS-B on JAX gradients,
#   - per-n-PE component accessor,
#   - Hessian-based 1-sigma errors.
#
# Subclasses provide the model-specific callables and parameter blocks.
# Flat parameter layout consumed by the optimiser:
#
#     theta = [log_A, *extra, *spe, lam]
# ===========================================================================

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from math import log
from scipy.optimize import minimize

from .fft_grid import build_grid, FFTGrid
from .likelihood import make_binned_logl, _spectrum_fft, _bin_integrals, density_on_xsp


# ==============================
#     Parameter block spec
# ==============================


@dataclass
class ParamBlock:
    name: str
    names: list
    init: np.ndarray
    bounds: list


# ==============================
#     Fit result container
# ==============================


@dataclass
class FitResult:
    converged: bool
    theta: np.ndarray
    theta_err: np.ndarray
    logl: float
    n_iter: int
    message: str
    layout: dict = field(default_factory=dict)

    def block(self, name):
        sl = self.layout[name]
        return self.theta[sl], self.theta_err[sl]


# ==============================
#     SpectrumFitter
# ==============================


class SpectrumFitter:
    """Binned PMT spectrum fitter with JAX autodiff + analytic bin integration."""

    def __init__(
        self,
        hist,
        bins,
        A=None,
        q_min=None,
        q_max=None,
        dq=None,
        extra_block: ParamBlock = None,
        spe_block: ParamBlock = None,
        lam_init: float = None,
        lam_bounds=(1e-6, 1e2),
        log_A_err: float = 0.05,
    ):
        hist = np.asarray(hist, dtype=float)
        bins = np.asarray(bins, dtype=float)
        A = int(A) if A is not None else int(hist.sum())

        if lam_init is None:
            occ_est = min(float(hist.sum()) / max(A, 1), 1.0 - 1e-9)
            lam_init = -log(1.0 - occ_est)

        self.grid = build_grid(hist, bins, A=A, q_min=q_min, q_max=q_max, dq=dq)

        self.extra_block = extra_block or self._default_extra_block()
        self.spe_block = spe_block or self._default_spe_block()

        log_A_init = log(A)
        init_parts = [
            np.array([log_A_init]),
            np.asarray(self.extra_block.init, dtype=float),
            np.asarray(self.spe_block.init, dtype=float),
            np.array([float(lam_init)]),
        ]
        bounds_parts = [
            [(log_A_init * (1 - log_A_err), log_A_init * (1 + log_A_err))],
            list(self.extra_block.bounds),
            list(self.spe_block.bounds),
            [tuple(lam_bounds)],
        ]
        self.init = np.concatenate(init_parts)
        self.bounds = sum(bounds_parts, [])

        n_extra = len(self.extra_block.init)
        n_spe = len(self.spe_block.init)
        self.layout = {
            "log_A": slice(0, 1),
            "extra": slice(1, 1 + n_extra),
            "spe": slice(1 + n_extra, 1 + n_extra + n_spe),
            "lam": slice(1 + n_extra + n_spe, 2 + n_extra + n_spe),
        }
        self.dof = len(self.init)
        self.param_names = (
            ["log_A"]
            + list(self.extra_block.names)
            + list(self.spe_block.names)
            + ["lam"]
        )

        pdf_extra, ser_ft, count_pgf, efficiency = self._model_callables()
        self._pdf_extra = pdf_extra
        self._ser_ft = ser_ft
        self._count_pgf = count_pgf

        self._logl_raw = make_binned_logl(self.grid, pdf_extra, ser_ft, count_pgf)
        self._logl_jit = jax.jit(self._logl_from_theta)
        self._grad_jit = jax.jit(jax.grad(self._logl_from_theta))

    # ==============================
    #     theta <-> blocks
    # ==============================

    def _unpack(self, theta):
        return (
            theta[self.layout["log_A"].start],
            theta[self.layout["extra"]],
            theta[self.layout["spe"]],
            theta[self.layout["lam"].start],
        )

    def _logl_from_theta(self, theta):
        log_A, extra, spe, lam = self._unpack(theta)
        return self._logl_raw(log_A, extra, spe, lam)

    def logl(self, theta):
        return float(self._logl_jit(jnp.asarray(theta, dtype=jnp.float64)))

    # ==============================
    #     MLE
    # ==============================

    def fit_mle(self, theta0=None, maxiter=500, verbose=False):
        if theta0 is None:
            theta0 = self.init.copy()
        theta0 = np.asarray(theta0, dtype=np.float64)

        self._logl_jit(jnp.asarray(theta0))
        self._grad_jit(jnp.asarray(theta0))

        def neg_logl(x):
            return -float(self._logl_jit(jnp.asarray(x, dtype=jnp.float64)))

        def neg_grad(x):
            return -np.array(self._grad_jit(jnp.asarray(x, dtype=jnp.float64)))

        res = minimize(
            neg_logl,
            theta0,
            jac=neg_grad,
            method="L-BFGS-B",
            bounds=self.bounds,
            options={"maxiter": maxiter, "disp": verbose},
        )

        theta_hat = np.asarray(res.x, dtype=np.float64)
        theta_err = self._hessian_errors(theta_hat)

        return FitResult(
            converged=bool(res.success),
            theta=theta_hat,
            theta_err=theta_err,
            logl=-float(res.fun),
            n_iter=int(res.nit),
            message=str(res.message),
            layout=self.layout,
        )

    def _hessian_errors(self, theta):
        try:
            H = np.asarray(jax.hessian(self._logl_from_theta)(jnp.asarray(theta)))
            cov = -np.linalg.inv(H)
            diag = np.diag(cov)
            diag = np.where(diag > 0, diag, np.nan)
            return np.sqrt(diag)
        except Exception:
            return np.full_like(theta, np.nan)

    # ==============================
    #     Expected counts & density
    # ==============================

    def estimate_bin_counts(self, theta):
        """Expected counts per bin via analytic Fourier bin integration."""
        log_A, extra, spe, lam = self._unpack(jnp.asarray(theta))
        G_tilde = _spectrum_fft(
            extra,
            spe,
            lam,
            jnp.asarray(self.grid.freq),
            jnp.asarray(self.grid.xsp),
            float(self.grid.xsp_width),
            int(self.grid.i_zero),
            self._pdf_extra,
            self._ser_ft,
            self._count_pgf,
        )
        bin_int = _bin_integrals(
            G_tilde,
            jnp.asarray(self.grid.bins),
            jnp.asarray(self.grid.freq),
            len(self.grid.xsp),
            float(self.grid.xsp_width),
        )
        return np.asarray(jnp.exp(log_A) * bin_int)

    def estimate_density(self, theta):
        """Full density G(q) on the xsp grid (for plotting)."""
        log_A, extra, spe, lam = self._unpack(jnp.asarray(theta))
        g = density_on_xsp(
            extra,
            spe,
            lam,
            jnp.asarray(self.grid.freq),
            float(self.grid.xsp_width),
            int(self.grid.i_zero),
            jnp.asarray(self.grid.xsp),
            self._pdf_extra,
            self._ser_ft,
            self._count_pgf,
        )
        return np.asarray(jnp.exp(log_A) * g)

    def estimate_component_counts(self, theta, n):
        """Expected counts for exactly n PE per bin."""
        n_pgf = self._single_n_pgf(n)
        log_A, extra, spe, lam = self._unpack(jnp.asarray(theta))
        G_tilde = _spectrum_fft(
            extra,
            spe,
            lam,
            jnp.asarray(self.grid.freq),
            jnp.asarray(self.grid.xsp),
            float(self.grid.xsp_width),
            int(self.grid.i_zero),
            self._pdf_extra,
            self._ser_ft,
            n_pgf,
        )
        bin_int = _bin_integrals(
            G_tilde,
            jnp.asarray(self.grid.bins),
            jnp.asarray(self.grid.freq),
            len(self.grid.xsp),
            float(self.grid.xsp_width),
        )
        return np.asarray(jnp.exp(log_A) * bin_int)

    # ==============================
    #     Subclass interface
    # ==============================

    def _model_callables(self):
        raise NotImplementedError

    def _default_extra_block(self) -> ParamBlock:
        raise NotImplementedError

    def _default_spe_block(self) -> ParamBlock:
        raise NotImplementedError

    def _single_n_pgf(self, n):
        raise NotImplementedError

    def get_gain(self, spe_args, kind="gm"):
        raise NotImplementedError

    def spe_report(self, spe_args) -> dict:
        raise NotImplementedError
