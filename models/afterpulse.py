"""models/afterpulse.py

Full SiPM charge-spectrum model extending GenTweedieFitter with a
NegBin (Geom_0) afterpulsing term.

Per-avalanche afterpulse count: A_j ~ Geom_0(rho)
  P(A_j = a) = (1 - rho) * rho^a,  a = 0, 1, 2, ...

The default model keeps each afterpulse at a fixed charge Q_ap.  The
Beta-charge variant below keeps the same afterpulse count law but replaces the
single-afterpulse charge delta function by Q = G X, X ~ Beta(2, b_ap).
Aggregated over n avalanches, M_n | N=n ~ NegBin(n, 1-rho), with

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


_BETA2_LAGUERRE_NODES, _BETA2_LAGUERRE_WEIGHTS = np.polynomial.laguerre.laggauss(64)
_BETA2_LAGUERRE_NODES = jnp.asarray(_BETA2_LAGUERRE_NODES)
_BETA2_LAGUERRE_WEIGHTS = jnp.asarray(_BETA2_LAGUERRE_WEIGHTS)


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


def _ser_ft_negbin_exp_ap(freq, spe):
    """Deprecated name kept for compatibility; use _ser_ft_negbin_beta_ap."""
    return _ser_ft_negbin_beta_ap(freq, spe)


def _beta2_shape_from_mean_fraction(mean_fraction):
    """Beta(2, b) shape b for a requested mean fraction E[X].

    For X ~ Beta(2, b), E[X] = 2 / (2 + b).  The tiny clipping only protects
    against exact floating-point endpoints; the fitter bounds keep this in the
    physical open interval during optimisation.
    """
    mean_fraction = jnp.clip(mean_fraction, 1e-12, 1.0 - 1e-12)
    return 2.0 * (1.0 - mean_fraction) / mean_fraction


def _beta2_charge_cf(freq, charge_scale, mean_fraction):
    """Characteristic function for Q = charge_scale * X, X ~ Beta(2, b).

    The physical model is:

      * release time is exponentially distributed;
      * early AP triggering is recovery-suppressed, giving the near-zero
        charge density f_X(x) ~ x;
      * the finite charge window bounds one AP to 0 < Q < G.

    The fitted mean fraction m = E[X] fixes b through b = 2(1-m)/m.  To keep the
    CF JAX-jittable and stable for large b, integrate with the transform
    y = b * (-log(1-x)):

        f_X(x) dx ∝ x (1-x)^(b-1) dx
                  = (1 - exp(-y/b)) exp(-y) dy / b.

    Fixed Gauss-Laguerre nodes then handle the exp(-y) weight.  The quadrature
    is normalised explicitly so phi(0) = 1 to roundoff.
    """
    b_shape = _beta2_shape_from_mean_fraction(mean_fraction)
    x = -jnp.expm1(-_BETA2_LAGUERRE_NODES / b_shape)
    weighted_x = _BETA2_LAGUERRE_WEIGHTS * x
    denom = jnp.sum(weighted_x)

    phase = jnp.exp(-1j * freq[..., None] * charge_scale * x)
    return jnp.sum(weighted_x * phase, axis=-1) / denom


def _ser_ft_negbin_beta_ap(freq, spe):
    """Effective single-avalanche CF with NegBin AP counts and Beta AP charge.

    The AP count for one prompt avalanche is still A ~ Geom_0(rho), so the AP
    contribution remains the PGF evaluated at the single-AP charge CF:

        AP_factor(w) = (1-rho) / (1 - rho * phi_Beta(w)).

    The one-AP charge is Q = G X with X ~ Beta(2, b_ap).  We preserve the
    existing mean-charge parameterisation by setting

        E[Q] = beta * (1-rho) * G,
        E[X] = beta * (1-rho),
        b_ap = 2 * (1 - E[X]) / E[X].

    Thus rho controls AP multiplicity while beta controls the mean charge of
    each AP, not another count-like degree of freedom.
    """
    a, b = spe[0], spe[1]
    spe_mean, spe_sigma = spe_from_reparam(a, b)
    alpha_gamma = (spe_mean / spe_sigma) ** 2
    theta_gamma = spe_mean / alpha_gamma
    f_tilde = (1.0 + 1j * theta_gamma * freq) ** (-alpha_gamma)

    rho = jnp.exp(spe[3])
    beta = jax.nn.sigmoid(spe[4])
    mean_fraction = beta * (1.0 - rho)
    phi_beta = _beta2_charge_cf(freq, spe_mean, mean_fraction)
    ap_factor = (1.0 - rho) / (1.0 - rho * phi_beta)

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

    @staticmethod
    def estimate_from_histogram(charges, counts):
        """Data-driven init from a binned charge histogram.

        Uses valley detection for occupancy and scipy peak-finding for SPE.

        Parameters
        ----------
        charges : array, bin centres in ADC
        counts  : array, bin counts (same length)

        Returns
        -------
        dict with keys: ped_mean, ped_sigma, spe_mean, spe_sigma, lam_est
        """
        import scipy.signal  # lazy import — scipy is not a hard fitter dep

        charges = np.asarray(charges, dtype=float)
        counts = np.asarray(counts, dtype=float)

        # 1. Pedestal peak in ±100 ADC
        ped_mask = (charges >= -100.0) & (charges <= 100.0)
        if ped_mask.sum() > 0:
            ped_mean = float(charges[ped_mask][int(np.argmax(counts[ped_mask]))])
        else:
            ped_mean = float(charges[int(np.argmax(counts))])

        # 2. Pedestal sigma (weighted std within ±25 ADC of peak)
        half = np.abs(charges - ped_mean) < 25
        if half.sum() > 3:
            w = counts[half] + 1e-10
            ped_sigma = max(
                float(np.sqrt(np.average((charges[half] - ped_mean) ** 2, weights=w))),
                1.0,
            )
        else:
            ped_sigma = 12.0

        # 3. Valley between 0PE and 1PE: minimum in [ped_mean+0.5σ, ped_mean+3.5σ].
        # Narrow range keeps us in the 0PE-1PE gap even when gain ≈ 4*ped_sigma.
        val_mask = (charges > ped_mean + 0.5 * ped_sigma) & (
            charges < ped_mean + 3.5 * ped_sigma
        )
        if val_mask.sum() > 0:
            valley_pos = float(charges[val_mask][int(np.argmin(counts[val_mask]))])
        else:
            valley_pos = ped_mean + 3.0 * ped_sigma

        # 4. Occupancy: lam = -log(1 - n_above_valley / n_total)
        n_total = float(counts.sum())
        above = float(counts[charges > valley_pos].sum())
        if n_total > 0 and above > 0:
            lam_est = float(-np.log(1.0 - min(above / n_total, 1.0 - 1e-9)))
        else:
            lam_est = 0.5

        # 5. SPE peaks above valley via scipy.signal.find_peaks (up to 4 peaks)
        spe_mask = charges > valley_pos
        dq = float(np.median(np.diff(charges)))
        if spe_mask.sum() > 5:
            sc = charges[spe_mask]
            sv = counts[spe_mask]
            min_prom = max(float(sv.max()) * 0.05, 3.0)
            min_dist = max(int(len(sv) // 8), 3)
            peaks, _ = scipy.signal.find_peaks(
                sv, prominence=min_prom, distance=min_dist
            )
            if len(peaks) >= 3:
                # ≥3 peaks: corrected median spacing (handles skipped PE peaks)
                spacings = np.diff(sc[peaks])
                min_sp = float(spacings.min())
                corrected = spacings / np.maximum(np.round(spacings / min_sp), 1)
                spe_mean = float(np.median(corrected))
                widths, *_ = scipy.signal.peak_widths(sv, peaks[:1], rel_height=0.5)
                spe_sigma = max(float(widths[0]) * dq / 2.355, spe_mean * 0.08, 2.0)
            elif len(peaks) >= 1:
                # 1-2 peaks: first peak above valley = 1PE
                spe_mean = float(sc[peaks[0]])
                widths, *_ = scipy.signal.peak_widths(sv, peaks[:1], rel_height=0.5)
                spe_sigma = max(float(widths[0]) * dq / 2.355, spe_mean * 0.08, 2.0)
            else:
                pk = int(np.argmax(sv))
                spe_mean = float(sc[pk])
                spe_sigma = max(spe_mean * 0.12, 2.0)
        else:
            spe_mean = 35.0
            spe_sigma = 6.0

        return {
            "ped_mean": ped_mean,
            "ped_sigma": ped_sigma,
            "spe_mean": spe_mean,
            "spe_sigma": spe_sigma,
            "lam_est": lam_est,
        }

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


# ==============================
#     NegBinBetaAPFitter
# ==============================


class NegBinBetaAPFitter(NegBinAPFitter):
    """Gen-Tweedie spectrum with NegBin AP counts and Beta(2,b) AP charge.

    Compared with the fixed-Q_ap model, this smooths the AP charge contribution
    across the 0--1 PE interval while preserving the same first moment:

        E[Q_AP] = beta * (1-rho) * G.

    The Beta shape a=2 is the early-recovery suppression f(Q) ~ Q.  The fitted
    mean fixes b, so the model adds no extra weakly identified AP parameter on
    top of rho and beta.

    Optional dark-count term
    ------------------------
    Pass ``dark_ft`` (from ``fitter.models.dark_count.make_dark_ft``) and a
    matching ``dark_block`` (from ``make_dark_block``) to enable the compound-
    Poisson dark pulse contribution D~(w) = exp[mu_dark*(phi_d1(w)-1)].
    """

    def __init__(self, *args, dark_ft=None, **kwargs):
        # Must be set before super().__init__ calls _model_callables
        self._dark_ft_opt = dark_ft
        super().__init__(*args, **kwargs)

    def _model_callables(self):
        return _ft_extra, _ser_ft_negbin_beta_ap, _count_pgf, getattr(self, "_dark_ft_opt", None)

    def spe_report(self, spe_args) -> dict:
        r = NegBinAPFitter.spe_report(self, spe_args)
        ap_charge_mean = r.pop("Q_ap")
        mean_fraction = ap_charge_mean / r["spe_mean"]
        r["ap_charge_mean"] = ap_charge_mean
        r["beta_shape_b"] = 2.0 * (1.0 - mean_fraction) / mean_fraction
        return r

    def spe_print(self, spe_args):
        r = self.spe_report(spe_args)
        mean_fraction = r["ap_charge_mean"] / r["spe_mean"]
        print(f"  spe    {'spe_mean':20s} = {r['spe_mean']:.6g}", flush=True)
        print(f"  spe    {'spe_sigma':20s} = {r['spe_sigma']:.6g}", flush=True)
        print(f"  spe    {'xi':20s} = {r['xi']:.6g}", flush=True)
        print(f"  spe    {'rho':20s} = {r['rho']:.6g}", flush=True)
        print(f"  spe    {'<Q_AP>/G':20s} = {mean_fraction:.6g}", flush=True)
        print(f"  spe    {'Beta shape b':20s} = {r['beta_shape_b']:.6g}", flush=True)


class NegBinExpAPFitter(NegBinBetaAPFitter):
    """Compatibility alias for the former exponential AP-charge fitter.

    The replacement model is NB AP counts with a bounded Beta(2,b) AP charge
    distribution.  Keep the old class name importable so older scripts do not
    break, but new code should use NegBinBetaAPFitter.
    """
