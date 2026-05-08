# ===========================================================================
# core/base.py
#
# Bundles for one spectrum:
#   - static FFT grid built from raw Q array (FD binning by default),
#   - JAX-jitted extended-Poisson log-likelihood — unbinned (NUFFT) by
#     default, binned as fallback when bins= is given,
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
from .likelihood import (
    make_binned_logl,
    make_unbinned_logl,
    _spectrum_fft,
    _bin_integrals,
    density_on_xsp,
)


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
    """PMT spectrum fitter with JAX autodiff.

    Accepts raw charge values Q_raw and builds the histogram internally.
    Fitting is unbinned by default (NUFFT density at each Q); pass
    mode="binned" to use the analytic Fourier bin integration instead.

    Parameters
    ----------
    Q_raw : array-like, shape (N_obs,)
        Raw charge values.
    A : int or None
        Total events including any not in Q_raw (e.g. pre-trigger).
        Defaults to len(Q_raw).
    bins : int | str | array-like | None
        Forwarded to numpy.histogram.  None uses the Freedman-Diaconis
        rule ("fd").
    q_min, q_max : float or None
        Override the FFT grid extent.
    dq : float or None
        FFT grid spacing.  Defaults to the histogram bin width.
    extra_block, spe_block : ParamBlock or None
        Parameter blocks; built via _default_* if None.
    lam_init : float or None
        Initial occupancy mean.  Estimated from data if None.
    lam_bounds : tuple
    log_A_err : float
        Fractional half-width of the log_A bound around log(A).
    mode : {"unbinned", "binned"}
    """

    def __init__(
        self,
        Q_raw,
        A=None,
        bins=None,
        q_min=None,
        q_max=None,
        dq=None,
        extra_block: ParamBlock = None,
        spe_block: ParamBlock = None,
        lam_init: float = None,
        lam_bounds=(1e-6, 1e2),
        lam_dc: float = 0.0,
        log_A_err: float = 0.05,
        mode: str = "unbinned",
    ):
        Q_raw = np.asarray(Q_raw, dtype=float)
        _A = A or len(Q_raw)

        # ===========================
        #     Internal histogram
        # ===========================
        _bins = "fd" if bins is None else bins
        hist, bin_edges = np.histogram(Q_raw, bins=_bins)

        if lam_init is None:
            occ_est = min(float(hist.sum()) / max(_A, 1), 1.0 - 1e-9)
            lam_init_hat = -log(1.0 - occ_est)
            lam_init_ = lam_init_hat - (np.exp(lam_init_hat) - 1) / (2 * _A)
            # a bit lower init somewhat helps with the converging
            lam_init_ *= 0.95
        else:
            lam_init_ = float(lam_init)

        self.lam_dc = float(lam_dc)
        self.Q_raw = Q_raw
        self.hist = hist
        self.bins = bin_edges
        self.mode = mode

        self.grid = build_grid(hist, bin_edges, A=_A, q_min=q_min, q_max=q_max, dq=dq)

        # ========================
        #     Parameter layout
        # ========================
        self.extra_block = extra_block or self._default_extra_block()
        self.spe_block = spe_block or self._default_spe_block()

        log_A_init = log(_A)
        init_parts = [
            np.array([log_A_init]),
            np.asarray(self.extra_block.init, dtype=float),
            np.asarray(self.spe_block.init, dtype=float),
            np.array([max(lam_init_ - self.lam_dc, 1e-3)]),
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

        # ==================
        #     Likelihood
        # ==================
        ft_extra, ser_ft, count_pgf, _ = self._model_callables()
        self._ft_extra = ft_extra
        self._ser_ft = ser_ft
        self._count_pgf = count_pgf

        if mode == "unbinned":
            self._logl_raw = make_unbinned_logl(
                jnp.asarray(Q_raw, dtype=jnp.float32),
                self.grid,
                ft_extra,
                ser_ft,
                count_pgf,
            )
        else:
            self._logl_raw = make_binned_logl(self.grid, ft_extra, ser_ft, count_pgf)

        self._logl_jit = jax.jit(self._logl_from_theta)
        self._grad_jit = jax.jit(jax.grad(self._logl_from_theta))

    # ==============================
    #     theta <-> blocks
    # ==============================

    def _unpack(self, theta):
        log_A = theta[self.layout["log_A"]]
        extra = theta[self.layout["extra"]]
        spe   = theta[self.layout["spe"]]
        lam   = theta[self.layout["lam"]] + self.lam_dc
        return log_A, extra, spe, lam

    def _logl_from_theta(self, theta):
        log_A, extra, spe, lam = self._unpack(theta)
        return self._logl_raw(log_A, extra, spe, lam)

    def logl(self, theta):
        return float(self._logl_jit(jnp.asarray(theta, dtype=jnp.float64)))

    # ===========
    #     MLE
    # ===========

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
            options={"maxiter": maxiter},
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

    def _G_tilde(self, theta):
        """Compute G_tilde from a theta array (JAX or numpy)."""
        _, extra, spe, lam = self._unpack(jnp.asarray(theta))
        return _spectrum_fft(
            extra,
            spe,
            lam,
            jnp.asarray(self.grid.freq),
            self._ft_extra,
            self._ser_ft,
            self._count_pgf,
        )

    def estimate_bin_counts(self, theta):
        """Expected counts per bin via analytic Fourier bin integration."""
        log_A, *_ = self._unpack(jnp.asarray(theta))
        G_tilde = self._G_tilde(theta)
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
        log_A, *_ = self._unpack(jnp.asarray(theta))
        G_tilde = self._G_tilde(theta)
        g = density_on_xsp(G_tilde, float(self.grid.xsp_width), int(self.grid.i_zero))
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
            self._ft_extra,
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
