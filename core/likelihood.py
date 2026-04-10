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
# g0~(w) is supplied analytically by ft_extra(freq, extra).
#
# Given G~_n on a uniform grid of length N and spacing dq (period L=N*dq),
# the inverse-DFT reconstruction is
#
#     G(q) = (1/L) * sum_n G~_n * exp(+i * w_n * q)
#
# and the antiderivative is
#
#     I(q) = (G~_0 / L) * q
#          + (1/L) * sum_{n!=0} (G~_n / (i*w_n)) * exp(+i*w_n*q)
#
# so bin integrals are I(b_{k+1}) - I(b_k), one matrix product over all edges.
#
# Model-specific pieces injected as JAX callables:
#     ft_extra(freq, extra)      -> g0~(w), analytic pedestal FT, shape (N,) complex
#     ser_ft(freq, spe)          -> f~(w), SPE characteristic function, shape (N,) complex
#     count_pgf(s, lam, spe)     -> pgf of count distribution at s
# ===========================================================================

import jax
import jax.numpy as jnp


# =================================
#     Fourier coefficients of G
# =================================


def _spectrum_fft(extra, spe, lam, freq, ft_extra, ser_ft, count_pgf):
    """Return G~_n on the uniform frequency grid.

    G~(w) = g0~(w) * pgf( f~(w) )
    """
    b_sp = ft_extra(freq, extra)
    s = ser_ft(freq, spe)
    return count_pgf(s, lam, spe) * b_sp


# ==============================
#     Analytic bin integrals
# ==============================


def _bin_integrals(G_tilde, edges, freq, N, dq):
    """Return integral_{edges[k]}^{edges[k+1]} G(q) dq for each k.

    Parameters
    ----------
    G_tilde : complex ndarray, shape (N,)
    edges   : ndarray, shape (nbin+1,)
    freq    : ndarray, shape (N,)   angular frequencies w_n = 2pi*fftfreq(N,dq)
    N, dq   : int, float
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


def make_binned_logl(grid, ft_extra, ser_ft, count_pgf, efficiency=None):
    """Build a binned extended-Poisson log-likelihood closure.

    Parameters
    ----------
    grid : FFTGrid
    ft_extra, ser_ft, count_pgf : JAX-compatible callables
    efficiency : ignored (kept for future threshold support)

    Returns
    -------
    logl : callable
        logl(log_A, extra, spe, lam, thres=None) -> scalar log-L.
    """
    freq = jnp.asarray(grid.freq)
    hist = jnp.asarray(grid.hist)
    edges = jnp.asarray(grid.bins)
    dq = float(grid.xsp_width)
    N = len(grid.xsp)
    zero = float(grid.zero)
    log_C = float(grid.log_C)

    def logl(log_A, extra, spe, lam, thres=None):
        A = jnp.exp(log_A)
        G_tilde = _spectrum_fft(extra, spe, lam, freq, ft_extra, ser_ft, count_pgf)

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
#     Unbinned log-likelihood
# ==============================


def _density_at_q(G_tilde, Q_raw, N, dq):
    """Evaluate G(q) at arbitrary positions via type-2 NUFFT.

    G(q) = (1/L) * sum_n G~_n * exp(+i w_n q),  w_n = 2pi/L * n.
    jax_finufft nufft2 expects modes in fftshift order and x in [-pi, pi].
    """
    import jax_finufft

    L = N * dq
    x = (2.0 * jnp.pi / L) * Q_raw.astype(jnp.float32)
    vals = jax_finufft.nufft2(
        jnp.fft.fftshift(G_tilde).astype(jnp.complex64), x, iflag=1
    )
    return jnp.maximum(jnp.real(vals) / L, 1e-32)


# def _density_at_q(G_tilde, Q_scalar, N, dq):
#     """Evaluate log G(q) at a single point via direct Fourier sum.

#     For a scalar Q this is cheaper and more vmap-friendly than NUFFT.
#     G(q) = (1/L) * Re( sum_n G~_n * exp(+i * w_n * q) )
#     """
#     import jax

#     L = N * dq
#     freq = (
#         2.0 * jnp.pi * jnp.fft.fftfreq(N, d=dq)
#     )  # already available as closed-over freq_j
#     phase = jnp.exp(1j * freq * Q_scalar.astype(jnp.float64))
#     G_val = jnp.real(jnp.dot(G_tilde, phase)) / L
#     return jnp.log(jnp.maximum(G_val, 1e-32))


def make_unbinned_logl(Q_raw, grid, ft_extra, ser_ft, count_pgf):
    """Build an unbinned extended-Poisson log-likelihood closure.

    G(q) is evaluated at each raw charge value via NUFFT (type-2:
    uniform Fourier coefficients -> non-uniform points), so no binning
    is needed for the shape term.

    Extended log-L:
        log L = N_obs*log(A) + sum_i log G(Q_i) - A * Re(G~_0)

    where Re(G~_0) = integral G dq ~ 1 when the grid covers the support.

    Parameters
    ----------
    Q_raw : array-like, shape (N_obs,)
        Raw charge values.
    grid : FFTGrid
        FFT grid built from the data range; hist/bins unused here.
    ft_extra, ser_ft, count_pgf : JAX callables
        Same model pieces as for make_binned_logl.

    Returns
    -------
    logl : callable
        logl(log_A, extra, spe, lam, thres=None) -> scalar.
    """
    freq = jnp.asarray(grid.freq)
    dq = float(grid.xsp_width)
    N = len(grid.xsp)
    N_obs = int(Q_raw.shape[0])
    Q = jnp.asarray(Q_raw, dtype=jnp.float32)

    def logl(log_A, extra, spe, lam, thres=None):
        A = jnp.exp(log_A)
        G_tilde = _spectrum_fft(extra, spe, lam, freq, ft_extra, ser_ft, count_pgf)
        G_vals = _density_at_q(G_tilde, Q, N, dq)
        p_total = jnp.real(G_tilde[0])
        return N_obs * jnp.log(A) + jnp.sum(jnp.log(G_vals)) - A * p_total

    return logl


# ============================
#     Density for plotting
# ============================


def density_on_xsp(G_tilde, xsp_width, i_zero):
    """Return G(q) on the uniform xsp grid via IFFT of precomputed G_tilde.

    Parameters
    ----------
    G_tilde   : complex ndarray, shape (N,)   precomputed by _spectrum_fft
    xsp_width : float                          grid spacing dq
    i_zero    : int                            index where xsp[i_zero] = 0

    Returns
    -------
    density : real ndarray, shape (N,)
    """
    density = jnp.real(jnp.fft.ifft(G_tilde)) / float(xsp_width)
    density = jnp.roll(density, int(i_zero))
    return jnp.maximum(density, 0.0)


def bin_integrals_for_theta(G_tilde, edges, freq, N, dq):
    """Public accessor for the analytic bin integrals (used by diagnostics)."""
    return _bin_integrals(G_tilde, edges, freq, N, dq)


# ==================================
#     Per-channel lam likelihood
# ==================================


def _count_u_from_pgf0(s, spe, count_pgf):
    """Return u(s) for count_pgf(s, lam, spe) = exp(lam * u(s, spe)).

    For the reconstruction path we use the fact that the count PGF is
    exponential-linear in lam. Since count_pgf(..., 0, ...) = 1, the
    derivative at lam = 0 equals u.
    """

    def _pgf_of_lam(lam_scalar):
        return count_pgf(s, lam_scalar, spe)

    return jax.jacfwd(_pgf_of_lam)(0.0)


# ==================================
#     Per-channel lam likelihood
# ==================================


def make_lam_logl(freq, dq, N, ft_extra, ser_ft, count_pgf):
    """Factory for the per-event reconstruction likelihood.

    Only fired channels are passed to this function — unfired channels
    contribute -lam_j to the NLL with trivial gradient, handled outside.

    The channel-calibration quantities are fixed across events, so they
    should be precomputed once via precompute_channels. For each event,
    only the observed charge Q enters through the Fourier phase factor;
    this is handled by precompute_event. During optimisation, only lam
    changes.

    Parameters
    ----------
    freq, dq, N             : shared grid geometry
    ft_extra, ser_ft,
    count_pgf               : JAX model callables

    Returns
    -------
    precompute_channels : callable
        precompute_channels(extra, spe) -> (s_all, g0_all, u_all)
        Called once for all active channels.
        extra: (n_ch, 2), spe: (n_ch, 3)
        s_all:  (n_ch, N) complex  -- SPE char function
        g0_all: (n_ch, N) complex  -- pedestal FT
        u_all:  (n_ch, N) complex  -- linear term in log pgf

    precompute_event : callable
        precompute_event(Q, fired_idx, g0_all, u_all) -> (u_sel, g0_phase)
        Called once per event for fired channels.
        Q: (n_fired,), fired_idx: (n_fired,) int32/int64
        u_sel:    (n_fired, N) complex
        g0_phase: (n_fired, N) complex

    logl_and_grad_fn : callable
        logl_and_grad_fn(lam, u_sel, g0_phase) -> (logl, grad)
        lam: (n_fired,), outputs both shape (n_fired,).
        Computes logl and d(logl)/d(lam_j) analytically in one pass.
    """
    freq_j = jnp.asarray(freq)
    L = float(N * dq)

    def precompute_channels(extra, spe):
        """Precompute channel-wise arrays independent of event and lam."""
        s_all = jax.vmap(lambda spe_j: ser_ft(freq_j, spe_j))(spe)  # (n_ch, N)
        g0_all = jax.vmap(lambda ex_j: ft_extra(freq_j, ex_j))(extra)  # (n_ch, N)
        u_all = jax.vmap(lambda s_j, spe_j: _count_u_from_pgf0(s_j, spe_j, count_pgf))(
            s_all, spe
        )  # (n_ch, N)
        return s_all, g0_all, u_all

    def precompute_event(Q, fired_idx, g0_all, u_all):
        """Precompute event-wise arrays for fired channels only.

        The observed charge enters only through exp(+i * w * Q_j), i.e. the
        Fourier kernel used to evaluate G_j at q = Q_j.
        """
        g0_sel = g0_all[fired_idx]  # (n_fired, N)
        u_sel = u_all[fired_idx]  # (n_fired, N)
        phase = jnp.exp(1j * jnp.outer(Q.astype(jnp.float64), freq_j))  # (n_fired, N)
        return u_sel, g0_sel * phase

    def logl_and_grad_fn(lam, u_sel, g0_phase):
        """Analytic logl and gradient w.r.t. lam for fired channels.

        G_j(Q_j; lam_j) = (1/L) * Re( sum_n exp(lam_j * u_jn) * g0_phase_jn )

        d(log G_j)/d(lam_j) = Re( sum_n u_jn * exp(lam_j * u_jn) * g0_phase_jn )
                               / (L * G_j)

        Both numerator and denominator share the same exponential matrix, so the
        gradient costs no extra FFT — one matrix product computes both.
        """
        expo = jnp.exp(lam[:, None] * u_sel)  # (n_fired, N)
        wt = expo * g0_phase  # (n_fired, N)
        G_val = jnp.real(jnp.sum(wt, axis=1)) / L  # (n_fired,)
        dG = jnp.real(jnp.sum(u_sel * wt, axis=1)) / L  # (n_fired,)
        G_safe = jnp.maximum(G_val, 1e-32)
        return jnp.log(G_safe), dG / G_safe

    precompute_channels_jit = jax.jit(precompute_channels)
    precompute_event_jit = jax.jit(precompute_event)
    logl_and_grad_jit = jax.jit(logl_and_grad_fn)

    return precompute_channels_jit, precompute_event_jit, logl_and_grad_jit
