"""models/afterpulse.py

Full SiPM charge-spectrum model extending GenTweedieFitter with a
NegBin (Geom_0) afterpulsing term.

Per-avalanche afterpulse count: A_j ~ Geom_0(rho)
  P(A_j = a) = (1 - rho) * rho^a,  a = 0, 1, 2, ...

Each afterpulse contributes a fixed charge Q_ap.  Aggregated over n
avalanches, M_n | N=n ~ NegBin(n, 1-rho), with

  E[M_n * Q_ap | N=n]   = n * Q_ap * rho / (1 - rho)
  Var[M_n * Q_ap | N=n] = n * (Q_ap / (1 - rho))^2 * rho

Parameterisation of afterpulse parameters
------------------------------------------
The two optimised parameters are:

  log_rho     log(rho),                  rho  = exp(log_rho)      in (0, 1)
  logit_beta  logit(beta),               beta = sigmoid(logit_beta) in (0, 1)

where beta = Q_ap / ((1 - rho) * G) and G = spe_mean, so that

  Q_ap = beta * (1 - rho) * G   in  (0, (1-rho)*G)  ⊂  (0, G)

Physical interpretation:
  - rho   controls the *mean* AP contribution: E ~ n * beta * G * rho
  - beta  controls the *spread*: Var ~ n * beta^2 * G^2 * rho
  These are approximately orthogonal in terms of the moments they govern.

  alpha = rho * beta = rho * Q_ap / ((1-rho) * G)  is a useful summary:
  it equals E[M_n * Q_ap | N=n] / (n * G * (1-rho)), i.e. the mean AP
  charge as a fraction of (1-rho)*G.

Using log_rho (rather than logit) keeps the gradient ~rho near zero,
which is more uniform for the expected regime rho ~ 0.1-1%.

Full spe_block layout (indices 0-4)
-------------------------------------
  0  a_logSigma   log(sigma_SPE)
  1  b_logDiff    log(mu_SPE - sigma_SPE)
  2  xi           Gen-Poisson dispersion
  3  log_rho      log(rho),   rho in (exp(-10), exp(-0.1)) ~ (4.5e-5, 0.9)
  4  logit_beta   logit(beta), beta in (sigmoid(-10), sigmoid(10)) ~ (5e-5, 1-5e-5)
"""

from __future__ import annotations

from math import log

import numpy as np
import jax.nn
import jax.numpy as jnp

from .gen_tweedie import (
    _ft_extra,
    _count_pgf,
    _make_single_n_pgf,
    GenTweedieFitter,
    reparam_from_spe,
    spe_from_reparam,
)
from ..core.base import ParamBlock


# ==============================
#     Effective SER with AP
# ==============================


def _ser_ft_negbin_ap(freq, spe):
    """Effective single-avalanche CF with NegBin afterpulsing.

    c_tilde(w) = f_tilde(w) * (1 - rho) / (1 - rho * exp(-i*w*Q_ap))

    Parameters
    ----------
    spe : array, length 5
        [a_logSigma, b_logDiff, xi, log_rho, logit_beta]

    Derived quantities
    ------------------
    rho  = exp(log_rho)                       in (0, 1)
    beta = sigmoid(logit_beta)                in (0, 1)
    Q_ap = beta * (1 - rho) * spe_mean        in (0, (1-rho)*G) ⊂ (0, G)
    """
    a, b = spe[0], spe[1]
    spe_mean, spe_sigma = spe_from_reparam(a, b)
    alpha_gamma = (spe_mean / spe_sigma) ** 2
    theta_gamma = spe_mean / alpha_gamma
    f_tilde = (1.0 + 1j * theta_gamma * freq) ** (-alpha_gamma)

    rho = jnp.exp(spe[3])
    beta = jax.nn.sigmoid(spe[4])
    Q_ap = beta * (1.0 - rho) * spe_mean

    ap_phase = jnp.exp(-1j * freq * Q_ap)
    ap_factor = (1.0 - rho) / (1.0 - rho * ap_phase)

    return f_tilde * ap_factor


# ==============================
#     Default parameter values
# ==============================

_DEFAULT_PED_MEAN = -600.0
_DEFAULT_PED_SIGMA = 600.0
_DEFAULT_SPE_MEAN = 6000.0
_DEFAULT_SPE_SIGMA = 800.0
_DEFAULT_XI = 0.04
_DEFAULT_RHO = 0.01  # ~1% afterpulse probability per avalanche
_DEFAULT_Q_AP = 2500.0  # afterpulse charge in ADC


# ==============================
#     NegBinAPFitter
# ==============================


class NegBinAPFitter(GenTweedieFitter):
    """Gen-Tweedie charge spectrum with NegBin (Geom_0) per-avalanche afterpulsing.

    Inherits all likelihood, optimiser, and plotting machinery from
    GenTweedieFitter.  Only _model_callables and the spe_block differ.

    See module docstring for the (log_rho, logit_beta) parameterisation.
    """

    def _model_callables(self):
        return _ft_extra, _ser_ft_negbin_ap, _count_pgf, None

    def _default_extra_block(self):
        return ParamBlock(
            name="pedestal",
            names=["ped_mean", "ped_sigma"],
            init=np.array([_DEFAULT_PED_MEAN, _DEFAULT_PED_SIGMA], dtype=float),
            bounds=[(-2000.0, 1500.0), (1.0, 2000.0)],
        )

    def _default_spe_block(self):
        a0, b0 = reparam_from_spe(_DEFAULT_SPE_MEAN, _DEFAULT_SPE_SIGMA)
        log_rho0 = float(np.log(_DEFAULT_RHO))
        beta_0 = _DEFAULT_Q_AP / ((1.0 - _DEFAULT_RHO) * _DEFAULT_SPE_MEAN)
        logit_beta0 = float(np.log(beta_0 / (1.0 - beta_0)))
        return ParamBlock(
            name="spe",
            names=["a_logSigma", "b_logDiff", "xi", "log_rho", "logit_beta"],
            init=np.array([a0, b0, _DEFAULT_XI, log_rho0, logit_beta0], dtype=float),
            bounds=[
                (log(10.0), log(1e5)),  # a_logSigma
                (log(10.0), log(1e5)),  # b_logDiff
                (1e-4, 0.99),  # xi
                (-10.0, -1.61),  # log_rho:    rho in (4.5e-5, 0.2)
                (-10.0, 10.0),  # logit_beta: beta in (5e-5, 1-5e-5)
            ],
        )

    def _single_n_pgf(self, n):
        return _make_single_n_pgf(n)

    def get_gain(self, spe_args, kind="gm"):
        a, b = float(spe_args[0]), float(spe_args[1])
        sigma = float(np.exp(a))
        mean = sigma + float(np.exp(b))
        rho = float(np.exp(float(spe_args[3])))
        beta = float(jax.nn.sigmoid(float(spe_args[4])))
        Q_ap = beta * (1.0 - rho) * mean
        mean_ap = rho / (1.0 - rho)  # E[A_j] for Geom_0(rho)
        if kind == "gm":
            return mean + mean_ap * Q_ap
        if kind == "prompt":
            return mean
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe_args) -> dict:
        a, b, xi = float(spe_args[0]), float(spe_args[1]), float(spe_args[2])
        sigma = float(np.exp(a))
        mean = sigma + float(np.exp(b))
        rho = float(np.exp(float(spe_args[3])))
        beta = float(jax.nn.sigmoid(float(spe_args[4])))
        Q_ap = beta * (1.0 - rho) * mean
        mean_ap = rho / (1.0 - rho)
        alpha = rho * beta
        return {
            "spe_mean": mean,
            "spe_sigma": sigma,
            "spe_res": sigma / mean,
            "xi": xi,
            "rho": rho,
            "beta": beta,
            "Q_ap": Q_ap,
            "alpha": alpha,
            "mean_ap": mean_ap,
            "total_mean": mean + mean_ap * Q_ap,
        }

    def spe_print(self, spe_args):
        r = self.spe_report(spe_args)
        print(f"  spe    {'spe_mean':20s} = {r['spe_mean']:.6g}", flush=True)
        print(f"  spe    {'spe_sigma':20s} = {r['spe_sigma']:.6g}", flush=True)
        print(f"  spe    {'xi':20s} = {r['xi']:.6g}", flush=True)
        print(f"  spe    {'rho':20s} = {r['rho']:.6g}", flush=True)
        print(f"  spe    {'beta=Q_ap/((1-rho)*G)':20s} = {r['beta']:.6g}", flush=True)
        print(f"  spe    {'Q_ap':20s} = {r['Q_ap']:.6g}", flush=True)
        print(f"  spe    {'alpha=rho*beta':20s} = {r['alpha']:.6g}", flush=True)
