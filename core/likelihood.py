# ===========================================================================
# core/likelihood.py
#
# JAX log-likelihood using analytic Fourier-domain bin integration.
#
# Model:
#     G(q) = g0(q) * sum_k  p_k(lam, ...) * f^{*k}(q)
# In Fourier space:
#     G~(w) = g0~(w) * pgf_N( f~(w); lam, ... )
#
# Given G~_n on a uniform grid of length N and spacing dq (so the period
# is L = N * dq), the inverse-DFT reconstruction is
#
#     G(q) = (1/L) * sum_n G~_n * exp(+i * w_n * q),
#
# with w_n = 2*pi * fftfreq(N, dq).  The antiderivative is
#
#     I(q) = (G~_0 / L) * q
#          + (1/L) * sum_{n != 0}  (G~_n / (i * w_n)) * exp(+i * w_n * q)
#
# so any bin integral is just  I(b_{k+1}) - I(b_k), computed once over
# all bin edges as a single matrix product.
#
# This replaces the previous Simpson-on-sub-sampled-grid scheme: no
# sub-sampling, no per-bin integration weights, no front/tail overflow
# weights — the overflow probability is simply  Re(G~_0) - sum_k bin_int_k.
#
# Model-specific pieces are injected as plain JAX callables:
#     pdf_extra(xsp, extra)      -> pedestal g0(q) on the uniform xsp grid
#     ser_ft(freq, spe)          -> SPE characteristic function f~(w)
#     count_pgf(s, lam, spe)     -> pgf of count distribution at s
# ===========================================================================

import jax.numpy as jnp


# ==============================
#     Fourier coefficients of G
# ==============================


def _spectrum_fft(extra, spe, lam, freq, xsp, dq, i_zero, pdf_extra, ser_ft, count_pgf):
    """Return G~_n on the uniform frequency grid.

    The pedestal is rolled so that q=0 sits at index 0 before the FFT,
    which makes the AC Fourier series reconstruct G(q) at the physical q
    (not q - xsp[0]).
    """
    g0 = pdf_extra(xsp, extra)
    g0_rolled = jnp.roll(g0, -i_zero)
    b_sp = jnp.fft.fft(g0_rolled) * dq  # g0~
    s = ser_ft(freq, spe)
    return count_pgf(s, lam, spe) * b_sp  # G~


# ==============================
#     Analytic bin integrals
# ==============================


def _bin_integrals(G_tilde, edges, freq, N, dq):
    """Return integral_{edges[k]}^{edges[k+1]} G(q) dq  for each k.

    Parameters
    ----------
    G_tilde : complex ndarray, length N
        Fourier coefficients of G on the periodic xsp grid (convention:
        G_tilde = fft(g_rolled) * dq, where g was rolled so q=0 -> index 0).
    edges : ndarray, length nbin+1
        Bin edges in physical q.
    freq : ndarray, length N
        Angular frequencies  w_n = 2*pi*fftfreq(N, dq).
    N : int
    dq : float
    """
    L = N * dq
    inv_L = 1.0 / L

    # AC part: coef_n = G~_n / (i w_n)  for n != 0;  coef_0 := 0.
    safe_w = jnp.where(freq == 0.0, 1.0, freq)
    coef = G_tilde / (1j * safe_w)
    coef = coef.at[0].set(0.0 + 0.0j)

    # phase matrix  exp(+i w_n * edges[k])    shape (nbin+1, N)
    phase = jnp.exp(1j * jnp.outer(edges, freq))

    I_ac = (phase @ coef) * inv_L  # (nbin+1,) complex
    I_dc = (G_tilde[0] * inv_L) * edges  # (nbin+1,) complex
    I = jnp.real(I_ac + I_dc)

    return I[1:] - I[:-1]  # (nbin,)


# ==============================
#     Binned log-likelihood
# ==============================


def make_binned_logl(grid, pdf_extra, ser_ft, count_pgf, efficiency=None):
    """Build a binned extended-Poisson log-likelihood closure.

    Parameters
    ----------
    grid : FFTGrid
    pdf_extra, ser_ft, count_pgf : JAX-compatible callables
    efficiency : ignored (kept for future threshold support)

    Returns
    -------
    logl : callable
        logl(log_A, extra, spe, lam, thres=None) -> scalar log-L.
    """
    xsp = jnp.asarray(grid.xsp)
    freq = jnp.asarray(grid.freq)
    hist = jnp.asarray(grid.hist)
    edges = jnp.asarray(grid.bins)
    dq = float(grid.xsp_width)
    i_zero = int(grid.i_zero)
    N = len(grid.xsp)
    zero = float(grid.zero)
    log_C = float(grid.log_C)

    def logl(log_A, extra, spe, lam, thres=None):
        A = jnp.exp(log_A)
        G_tilde = _spectrum_fft(
            extra,
            spe,
            lam,
            freq,
            xsp,
            dq,
            i_zero,
            pdf_extra,
            ser_ft,
            count_pgf,
        )

        bin_int = _bin_integrals(G_tilde, edges, freq, N, dq)
        y_est = jnp.maximum(A * bin_int, 1e-32)

        # overflow = A * (total probability - probability inside window)
        p_total = jnp.real(G_tilde[0])  # ~1 when grid covers support
        p_window = jnp.sum(bin_int)
        z_est = jnp.maximum(A * (p_total - p_window), 1e-32)

        ll_bins = jnp.sum(hist * jnp.log(y_est) - y_est)
        ll_zero = zero * jnp.log(z_est) - z_est
        return ll_bins + ll_zero - log_C

    return logl


# ==============================
#     Density for plotting / diagnostics
# ==============================


def density_on_xsp(
    extra, spe, lam, freq, xsp_width, i_zero, xsp, pdf_extra, ser_ft, count_pgf
):
    """Return G(q) on the uniform xsp grid via inverse FFT.

    Only used for plotting — the likelihood never needs this.
    """
    G_tilde = _spectrum_fft(
        extra,
        spe,
        lam,
        freq,
        xsp,
        float(xsp_width),
        int(i_zero),
        pdf_extra,
        ser_ft,
        count_pgf,
    )
    density = jnp.real(jnp.fft.ifft(G_tilde)) / float(xsp_width)
    density = jnp.roll(density, int(i_zero))
    return jnp.maximum(density, 0.0)


def bin_integrals_for_theta(G_tilde, edges, freq, N, dq):
    """Public accessor for the analytic bin integrals (used by diagnostics)."""
    return _bin_integrals(G_tilde, edges, freq, N, dq)
