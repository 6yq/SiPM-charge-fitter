# ===========================================================================
# models/gen_tweedie.py
#
# Compound-Generalized-Poisson-Gamma PMT model, JAX edition.
#
# Physical model:
#   G(q)  = g0(q) * sum_k  p_k(lam, xi) * f^{*k}(q)
#   G~(w) = g0~(w) * exp( lam * (T(f~(w)) - 1) )
#
# Pedestal:
#   g0(q) = N(ped_mean, ped_sigma^2)
#
# SPE (Gamma):
#   f~(w) = (1 + i * theta * w)^(-alpha)
#   alpha = (spe_mean / spe_sigma)^2,  theta = spe_mean / alpha
#
# Generalized-Poisson count distribution, parametrised by (lam, xi):
#   pgf( s ) = exp( lam * (T(s) - 1) )
#   T(s) solves  T = s * exp(xi * (T - 1))
#   => T(s) = -W( -xi * s * exp(-xi) ) / xi
#
# Reparameterisation:
#   a = log(spe_sigma)
#   b = log(spe_mean - spe_sigma)     (log of the "diff")
#   => spe_sigma = exp(a)
#      spe_mean  = exp(a) + exp(b)
# This automatically enforces spe_mean > spe_sigma (so alpha > 1 and the
# Gamma density has a mode strictly above zero).
# ===========================================================================

import numpy as np
import jax.numpy as jnp

from math import log

from ..core.base import SpectrumFitter, ParamBlock
from ..core.lambert_w import lambert_w0


# ==========================
#     Reparameterisation
# ==========================


def spe_from_reparam(a, b):
    """Return (spe_mean, spe_sigma) from (a, b) = (log sigma, log(mean-sigma))."""
    sigma = jnp.exp(a)
    mean = sigma + jnp.exp(b)
    return mean, sigma


def reparam_from_spe(spe_mean, spe_sigma):
    """Inverse of spe_from_reparam, for setting initial values."""
    assert spe_mean > spe_sigma, "spe_mean must exceed spe_sigma for alpha > 1"
    return log(spe_sigma), log(spe_mean - spe_sigma)


# =====================
#     JAX callables
# =====================


def _ft_extra(freq, extra):
    """Analytic DFT of the Gaussian pedestal, matching numpy.fft convention.
    g0~(w) = exp(-i*mu*w - sigma^2*w^2/2)
    """
    ped_mean, ped_sigma = extra[0], extra[1]
    return jnp.exp(-1j * ped_mean * freq - 0.5 * ped_sigma**2 * freq**2)


def _ser_ft(freq, spe):
    """Gamma characteristic function.  spe = (a, b, xi)."""
    a, b = spe[0], spe[1]
    spe_mean, spe_sigma = spe_from_reparam(a, b)
    alpha = (spe_mean / spe_sigma) ** 2
    theta = spe_mean / alpha
    return (1.0 + 1j * theta * freq) ** (-alpha)


def _count_pgf(s, lam, spe):
    """Gen-Poisson pgf at s:  exp( lam * (T(s) - 1) )."""
    xi = spe[2]
    xi_safe = jnp.where(jnp.abs(xi) < 1e-12, 1e-12, xi)
    arg = -xi_safe * s * jnp.exp(-xi_safe)
    T = -lambert_w0(arg) / xi_safe
    return jnp.exp(lam * (T - 1.0))


def _count_pmf(lam, xi, n):
    """Gen-Poisson PMF p_n(lam, xi).  n is a Python int at build time."""
    if n == 0:
        return jnp.exp(-lam)
    log_p = (
        jnp.log(lam)
        + (n - 1) * jnp.log(lam + xi * n)
        - lam
        - xi * n
        - float(np.sum(np.log(np.arange(1, n + 1))))
    )
    return jnp.exp(log_p)


def _make_single_n_pgf(n):
    """Build a count_pgf-compatible closure that returns p_n(lam,xi) * s^n."""

    def pgf(s, lam, spe):
        xi = spe[2]
        p_n = _count_pmf(lam, xi, n)
        return p_n * s**n

    return pgf


# ========================
#     GenTweedieFitter
# ========================


class GenTweedieFitter(SpectrumFitter):
    """Compound Generalized-Poisson / Gamma fitter with Gaussian pedestal."""

    # default physical initial values
    _DEFAULT_PED_MEAN = -600.0
    _DEFAULT_PED_SIGMA = 600.0
    _DEFAULT_SPE_MEAN = 6000.0
    _DEFAULT_SPE_SIGMA = 800.0
    _DEFAULT_XI = 0.04

    def _model_callables(self):
        return _ft_extra, _ser_ft, _count_pgf, None

    def _default_extra_block(self) -> ParamBlock:
        pm = self._DEFAULT_PED_MEAN
        ps = self._DEFAULT_PED_SIGMA
        return ParamBlock(
            name="pedestal",
            names=["ped_mean", "ped_sigma"],
            init=np.array([pm, ps], dtype=float),
            bounds=[(-2000.0, 1500.0), (1.0, 2000.0)],
        )

    def _default_spe_block(self) -> ParamBlock:
        a0, b0 = reparam_from_spe(self._DEFAULT_SPE_MEAN, self._DEFAULT_SPE_SIGMA)
        return ParamBlock(
            name="spe",
            names=["a_logSigma", "b_logDiff", "xi"],
            init=np.array([a0, b0, self._DEFAULT_XI], dtype=float),
            bounds=[
                (log(10.0), log(1e5)),  # a = log(sigma); sigma in [10, 1e5]
                (log(10.0), log(1e5)),  # b = log(diff); diff in [10, 1e5]
                (1e-4, 0.99),  # xi
            ],
        )

    def _single_n_pgf(self, n):
        return _make_single_n_pgf(n)

    def get_gain(self, spe_args, kind="gm"):
        a, b, _xi = float(spe_args[0]), float(spe_args[1]), float(spe_args[2])
        sigma = float(np.exp(a))
        mean = sigma + float(np.exp(b))
        alpha = (mean / sigma) ** 2
        theta = mean / alpha
        if kind == "gm":
            return mean
        if kind == "gp":
            return (alpha - 1.0) * theta
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe_args) -> dict:
        a, b, xi = float(spe_args[0]), float(spe_args[1]), float(spe_args[2])
        sigma = float(np.exp(a))
        mean = sigma + float(np.exp(b))
        return {
            "spe_mean": mean,
            "spe_sigma": sigma,
            "spe_res": sigma / mean,
            "xi": xi,
            "a": a,
            "b": b,
        }
