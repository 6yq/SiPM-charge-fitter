"""Dark count (DCR) term for SiPM charge spectra.

Compound-Poisson dark pulse contribution in frequency domain:

    D~(w) = exp[ mu_dark * (phi_d1(w) - 1) ]

where mu_dark = DCR * (T_gate + t0_pre) and phi_d1(w) is the single-dark-pulse
charge CF computed by Gauss-Legendre quadrature over the integration window.

With SPE fluctuations ignored (deterministic unit charge):

    phi_d1(w) = [1/(T+t0)] * integral_{-t0}^{T} exp(-i*w*G*a(s)) ds

where a(s) is the fractional recovered charge at time s:

    a(s) = exp(s/tau) * (1 - exp(-T/tau))    s in (-t0, 0)    [pre-gate]
    a(s) = 1 - exp(-(T-s)/tau)               s in [0, T)      [in-gate]

G* is extracted from spe_params = [a_logSigma, b_logDiff, ...].

Free parameter:  log_mu_dark = log(mu_dark)
Fixed externals: T_gate, t0_pre, tau_slow (all in the same time unit)
"""

from __future__ import annotations

from math import log

import numpy as np
import jax.numpy as jnp

from ..core.base import ParamBlock


def make_dark_ft(
    T_gate: float,
    t0_pre: float,
    tau_slow: float,
    n_quad: int = 64,
):
    """Build a JAX-jittable dark-count CF callable D~(w).

    Parameters
    ----------
    T_gate   : gate length (same unit as tau_slow, e.g. ns)
    t0_pre   : pre-gate window
    tau_slow : slow exponential pulse time constant
    n_quad   : Gauss-Legendre quadrature points per segment (pre + in-gate)

    Returns
    -------
    dark_ft : callable(freq, dark_params, spe_params) -> D~(w), shape (N,) complex
        dark_params[0] = log_mu_dark
        spe_params[0] = a_logSigma, spe_params[1] = b_logDiff  (for G*)
    """
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)

    # Pre-gate segment: s in [-t0_pre, 0]
    s_pre = 0.5 * t0_pre * (gl_nodes - 1.0)
    w_pre = 0.5 * t0_pre * gl_weights
    a_pre = np.exp(s_pre / tau_slow) * (1.0 - np.exp(-T_gate / tau_slow))

    # In-gate segment: s in [0, T_gate]
    s_in = 0.5 * T_gate * (gl_nodes + 1.0)
    w_in = 0.5 * T_gate * gl_weights
    a_in = 1.0 - np.exp(-(T_gate - s_in) / tau_slow)

    a_all = jnp.asarray(np.concatenate([a_pre, a_in]), dtype=jnp.float64)
    w_all = jnp.asarray(np.concatenate([w_pre, w_in]), dtype=jnp.float64)
    norm = float(T_gate + t0_pre)

    def dark_ft(freq, dark_params, spe_params):
        mu_dark = jnp.exp(dark_params[0])
        # G* from spe reparameterisation
        spe_sigma = jnp.exp(spe_params[0])
        spe_mean = spe_sigma + jnp.exp(spe_params[1])
        # phi_d1(w) via quadrature over fractional-charge knots a_all
        # phase[j, k] = exp(-i * freq[j] * G* * a_all[k])
        phase = jnp.exp(-1j * jnp.outer(freq, a_all) * spe_mean)  # (N, 2*n_quad)
        phi_d1 = jnp.sum(phase * w_all, axis=-1) / norm            # (N,)
        return jnp.exp(mu_dark * (phi_d1 - 1.0))

    return dark_ft


def make_dark_block(
    mu_dark_init: float = 0.1,
    mu_dark_lo: float = 1e-4,
    mu_dark_hi: float = 50.0,
) -> ParamBlock:
    """Parameter block for the single dark-count free parameter log_mu_dark."""
    return ParamBlock(
        name="dark",
        names=["log_mu_dark"],
        init=np.array([log(float(mu_dark_init))], dtype=float),
        bounds=[(log(float(mu_dark_lo)), log(float(mu_dark_hi)))],
    )
