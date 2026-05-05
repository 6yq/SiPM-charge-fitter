# ===========================================================================
# core/lambert_w.py
#
# JAX implementation of the principal branch of the Lambert-W function,
# W(z) satisfying W(z) * exp(W(z)) = z.
#
# Two backends, selected at import time:
#
#   TABLE (default):
#     Bilinear interpolation on a precomputed (Re, Im) grid loaded from
#     lambert_w_table.npz.  The table is built once by gen_lambert_w_table.py
#     and lives alongside this file.  Load cost at import is negligible
#     (~5 ms for a 1024×1024 npz).  Interpolation cost is O(1) per element
#     instead of 6 Halley iterations, giving a substantial speedup for
#     the vectorised (N,)-shaped inputs in precompute_channels.
#
#   HALLEY (fallback):
#     6 Halley iterations via lax.fori_loop, used automatically when the
#     table file is not found or when the query point falls outside the
#     table range.
#
# The custom JVP (dW/dz = W / (z * (1 + W))) is registered once and applies
# to both backends — the primal value is computed by whichever backend is
# active, then the analytic formula produces the tangent.
#
# To regenerate the table:
#   python3 gen_lambert_w_table.py -o fitter/core/lambert_w_table.npz --n 1024
# ===========================================================================

import os

import jax
import numpy as np
import jax.numpy as jnp


# ================================================================
#     Halley iteration backend (always available, used as fallback)
# ================================================================


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


def _lambert_w_iterate(z, n_iter=4):
    """Run a fixed number of Halley iterations.

    12 iterations is overkill for double precision (Halley converges cubically
    so 4-5 are usually enough) but keeps us safe near the branch point.

    Notes
    -----
    2026.4.8:
        Let's take 6 iterations then.
    2026.5.5:
        3 iterations already close enough to 12.
        Keep 4 for safety.
    """
    z = z.astype(jnp.complex128)
    w = _initial_guess(z)

    def body(i, w):
        return _halley_step(w, z)

    return jax.lax.fori_loop(0, n_iter, body, w)


# ================================================================
#     Table backend
#
#     Bilinear interpolation on a uniform (Re, Im) grid.
#     The table is loaded once at module import time.
#     All JAX ops are elementwise so the function is vmap/jit-compatible
#     and works on any shape input.
#
#     Interpolation formula for a point (re, im) in cell (i, j):
#       t = (re - re_pts[i]) / dre,   u = (im - im_pts[j]) / dim
#       W ≈ (1-t)(1-u) W[i,j] + t(1-u) W[i+1,j]
#           + (1-t)u W[i,j+1] + tu W[i+1,j+1]
#     applied independently to W_re and W_im.
# ================================================================

_TABLE_PATH = os.path.join(os.path.dirname(__file__), "lambert_w_table.npz")
_TABLE_LOADED = False
_re_pts_j = _im_pts_j = _W_re_j = _W_im_j = None
_re_lo = _re_hi = _im_lo = _im_hi = _dre = _dim = None


def _load_table():
    global _TABLE_LOADED
    global _re_pts_j, _im_pts_j, _W_re_j, _W_im_j
    global _re_lo, _re_hi, _im_lo, _im_hi, _dre, _dim

    if not os.path.exists(_TABLE_PATH):
        print(
            "[*] No Lambert-W table found, consider running `gen_lambert_w_table.py`",
            flush=True,
        )
        return False

    print("[*] Loading Lambert-W table", flush=True)
    data = np.load(_TABLE_PATH)
    _re_pts_j = jnp.asarray(data["re_pts"])
    _im_pts_j = jnp.asarray(data["im_pts"])
    _W_re_j = jnp.asarray(data["W_re"])
    _W_im_j = jnp.asarray(data["W_im"])

    _re_lo, _re_hi = float(_re_pts_j[0]), float(_re_pts_j[-1])
    _im_lo, _im_hi = float(_im_pts_j[0]), float(_im_pts_j[-1])
    _dre = float(_re_pts_j[1] - _re_pts_j[0])
    _dim = float(_im_pts_j[1] - _im_pts_j[0])

    _TABLE_LOADED = True
    return True


_load_table()


def _lambert_w_table(z):
    """Bilinear table lookup for Lambert W.  Same API as _lambert_w_iterate."""
    z = z.astype(jnp.complex128)
    re = jnp.real(z)
    im = jnp.imag(z)

    n_re = _W_re_j.shape[0]
    n_im = _W_re_j.shape[1]

    # Continuous index
    i_f = (re - _re_lo) / _dre
    j_f = (im - _im_lo) / _dim

    # Floor indices, clamped to valid range
    i0 = jnp.clip(jnp.floor(i_f).astype(jnp.int32), 0, n_re - 2)
    j0 = jnp.clip(jnp.floor(j_f).astype(jnp.int32), 0, n_im - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    # Fractional parts
    t = jnp.clip(i_f - i0.astype(jnp.float64), 0.0, 1.0)
    u = jnp.clip(j_f - j0.astype(jnp.float64), 0.0, 1.0)

    # Bilinear interpolation — real part
    wr = (
        (1 - t) * (1 - u) * _W_re_j[i0, j0]
        + t * (1 - u) * _W_re_j[i1, j0]
        + (1 - t) * u * _W_re_j[i0, j1]
        + t * u * _W_re_j[i1, j1]
    )
    # Bilinear interpolation — imaginary part
    wi = (
        (1 - t) * (1 - u) * _W_im_j[i0, j0]
        + t * (1 - u) * _W_im_j[i1, j0]
        + (1 - t) * u * _W_im_j[i0, j1]
        + t * u * _W_im_j[i1, j1]
    )

    return (wr + 1j * wi).astype(jnp.complex128)


# ================================================================
#     Public API: lambert_w0 with custom JVP
#
#     The primal uses the table when available, Halley otherwise.
#     The JVP is always the analytic formula dW/dz = W / (z*(1+W)),
#     independent of which primal backend is active.
# ================================================================


@jax.custom_jvp
def lambert_w0(z):
    """Principal-branch Lambert-W.

    Accepts real or complex input; output is always complex128 so that
    the downstream Fourier arithmetic stays in complex space.
    """
    if _TABLE_LOADED:
        return _lambert_w_table(z)
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
