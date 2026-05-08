"""models/afterpulse.py

Full SiPM charge-spectrum model extending GenTweedieFitter with a
NegBin (Geom_0) afterpulsing term.

Per-avalanche afterpulse count: A_j ~ Geom_0(rho)
  P(A_j = a) = (1 - rho) * rho^a,  a = 0, 1, 2, ...

Each afterpulse contributes a fixed charge Q_ap.  Aggregated over n
avalanches, M_n | N=n ~ NegBin(n, 1-rho), which captures over-dispersion
relative to a Poisson afterpulse model.

The modification enters only through the effective single-avalanche
characteristic function

  c_tilde(w) = f_tilde(w) * (1 - rho) / (1 - rho * exp(i*w*Q_ap))

Everything else (pedestal, Gen-Poisson count PGF, binned likelihood,
optimiser) is inherited unchanged from SpectrumFitter / GenTweedieFitter.

Parameter layout of spe_block
------------------------------
Indices 0-2: same as GenTweedieFitter
  0  a_logSigma   log(sigma_SPE)
  1  b_logDiff    log(mu_SPE - sigma_SPE)
  2  xi           Gen-Poisson dispersion

Indices 3-4 (afterpulse):
  3  logit_rho    logit(rho), rho in (0,1) via jax.nn.sigmoid
  4  log_Q_ap     log(Q_ap), afterpulse charge
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

    c_tilde(w) = f_tilde(w) * (1 - rho) / (1 - rho * exp(i*w*Q_ap))
    """
    a, b = spe[0], spe[1]
    spe_mean, spe_sigma = spe_from_reparam(a, b)
    alpha = (spe_mean / spe_sigma) ** 2
    theta = spe_mean / alpha
    f_tilde = (1.0 + 1j * theta * freq) ** (-alpha)

    rho = jax.nn.sigmoid(spe[3])
    Q_ap = jnp.exp(spe[4])
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
_DEFAULT_RHO = 0.05
_DEFAULT_Q_AP = 6000.0


# ==============================
#     NegBinAPFitter
# ==============================


class NegBinAPFitter(GenTweedieFitter):
    """Gen-Tweedie charge spectrum with NegBin (Geom_0) per-avalanche afterpulsing.

    Inherits all likelihood, optimiser, and plotting machinery from
    GenTweedieFitter.  Only _model_callables and the spe_block differ.
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
        # logit(0.05) = log(0.05 / 0.95)
        logit_rho0 = float(np.log(_DEFAULT_RHO / (1.0 - _DEFAULT_RHO)))
        return ParamBlock(
            name="spe",
            names=["a_logSigma", "b_logDiff", "xi", "logit_rho", "log_Q_ap"],
            init=np.array(
                [a0, b0, _DEFAULT_XI, logit_rho0, log(_DEFAULT_Q_AP)], dtype=float
            ),
            bounds=[
                (log(10.0), log(1e5)),  # a_logSigma
                (log(10.0), log(1e5)),  # b_logDiff
                (1e-4, 0.99),  # xi
                (
                    -10.0,
                    4.0,
                ),  # logit_rho: rho in (sigmoid(-10), sigmoid(4)) ≈ (5e-5, 0.98)
                (log(100.0), log(1e5)),  # log_Q_ap
            ],
        )

    def _single_n_pgf(self, n):
        return _make_single_n_pgf(n)

    def get_gain(self, spe_args, kind="gm"):
        a, b = float(spe_args[0]), float(spe_args[1])
        sigma = float(np.exp(a))
        mean = sigma + float(np.exp(b))
        rho = float(jax.nn.sigmoid(float(spe_args[3])))
        Q_ap = float(np.exp(spe_args[4]))
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
        rho = float(jax.nn.sigmoid(float(spe_args[3])))
        Q_ap = float(np.exp(spe_args[4]))
        mean_ap = rho / (1.0 - rho)
        return {
            "spe_mean": mean,
            "spe_sigma": sigma,
            "spe_res": sigma / mean,
            "xi": xi,
            "rho": rho,
            "Q_ap": Q_ap,
            "mean_ap": mean_ap,
            "total_mean": mean + mean_ap * Q_ap,
            "a": a,
            "b": b,
        }
