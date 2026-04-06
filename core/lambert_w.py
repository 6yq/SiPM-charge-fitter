# ===========================================================================
# core/lambert_w.py
#
# JAX implementation of the principal branch of the Lambert-W function,
# W(z) satisfying W(z) * exp(W(z)) = z.
#
# Works for complex z (the Fourier-domain argument of the Gen-Poisson pgf).
# Halley iteration is used; the analytic derivative
#     dW/dz = W / (z * (1 + W))
# is registered via jax.custom_jvp so that autodiff bypasses the iteration.
# ===========================================================================

import jax
import jax.numpy as jnp


# ==============================
#     Halley iteration kernel
# ==============================


def _initial_guess(z):
    """Initial guess for the principal-branch Lambert-W, valid on C.

    Strategy: use the Taylor series near the origin (where the asymptotic
    expansion diverges) and log(z) - log(log(z)) for |z| large enough that
    |log(z)| is safely non-zero.  In the intermediate band we start from
    a fixed real value on the principal branch.
    """
    z = z.astype(jnp.complex128)
    abs_z = jnp.abs(z)

    # Taylor series around z=0 (5 terms; good for |z| < ~0.3)
    series = z * (1.0 - z * (1.0 - z * (1.5 - z * (8.0 / 3.0))))

    # asymptotic: valid once |log(z)| is safely non-zero
    # substitute z=1.0+0j where it would be singular (our mask handles it)
    z_safe = jnp.where(abs_z < 0.1, 1.0 + 0j, z)
    log_z = jnp.log(z_safe)
    log_z_safe = jnp.where(jnp.abs(log_z) < 0.5, 1.0 + 0j, log_z)
    asymp = log_z - jnp.log(log_z_safe)

    # intermediate region: start from w=0.5 (real, on principal branch;
    # complex perturbations follow)
    intermediate = jnp.full_like(z, 0.5 + 0j)

    # pick by magnitude of z
    guess = jnp.where(abs_z < 0.3, series, intermediate)
    guess = jnp.where(abs_z > 2.0, asymp, guess)
    return guess


def _halley_step(w, z):
    """One Halley iteration for f(w) = w*exp(w) - z."""
    ew = jnp.exp(w)
    f = w * ew - z
    fp = ew * (1.0 + w)
    # Halley correction: denom = fp - f * f'' / (2 fp)
    # f'' = ew * (2 + w), so f * f'' / (2 fp) = f * (2 + w) / (2 * (1 + w))
    denom = fp - f * (2.0 + w) / (2.0 * (1.0 + w))
    return w - f / denom


def _lambert_w_iterate(z, n_iter=12):
    """Run a fixed number of Halley iterations.

    12 iterations is overkill for double precision (Halley converges cubically
    so 4–5 are usually enough) but keeps us safe near the branch point.
    """
    z = z.astype(jnp.complex128)
    w = _initial_guess(z)

    def body(i, w):
        return _halley_step(w, z)

    return jax.lax.fori_loop(0, n_iter, body, w)


# ==============================
#     Custom-JVP wrapper
# ==============================


@jax.custom_jvp
def lambert_w0(z):
    """Principal-branch Lambert-W.

    Accepts real or complex input; output is always complex128 so that
    the downstream Fourier arithmetic stays in complex space.
    """
    return _lambert_w_iterate(z)


@lambert_w0.defjvp
def _lambert_w0_jvp(primals, tangents):
    (z,) = primals
    (dz,) = tangents
    w = lambert_w0(z)
    # dW/dz = W / (z * (1 + W))
    # Near z=0, W=0, so W/z -> 1 and the full expression -> 1.
    safe = jnp.abs(z) < 1e-30
    z_safe = jnp.where(safe, 1.0 + 0j, z.astype(jnp.complex128))
    dw_dz = jnp.where(safe, 1.0 + 0j, w / (z_safe * (1.0 + w)))
    return w, dw_dz * dz.astype(jnp.complex128)
