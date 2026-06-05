"""Numerical-equivalence test: half-spectrum vs full-grid reference.

The half-spectrum path uses freq[:N//2+1] and a herm_w vector to reproduce
Re(Σ wt) exactly. This test builds both pipelines on the same small grid
with the PRODUCTION callables, and asserts the logl and grad agree at rtol
1e-5 on the float64 return values. The f32 cast is identical in both
branches so it cancels — the test isolates the Hermitian reduction.
"""

import jax
import jax.numpy as jnp
import numpy as np

from fitter.core.fft_grid import build_grid
from fitter.core.likelihood import make_lam_logl
from fitter.models.gen_tweedie import _ft_extra, _count_pgf, reparam_from_spe
from fitter.models.afterpulse import _ser_ft_negbin_beta_ap

jax.config.update("jax_enable_x64", True)

N_CH = 5


def _reference_full(lam, u_full, g0_phase_full, L):
    """Inline full-grid logl_and_grad for comparison, matching make_lam_logl's
    f32 hot-path + f64-accumulation pattern exactly."""
    lam32 = lam.astype(jnp.float32)
    u32 = u_full.astype(jnp.complex64)
    g32 = g0_phase_full.astype(jnp.complex64)
    expo = jnp.exp(lam32[:, None] * u32)
    wt = expo * g32
    G_val = jnp.real(jnp.sum(wt, axis=1)).astype(jnp.float64) / L
    dG = jnp.real(jnp.sum(u32 * wt, axis=1)).astype(jnp.float64) / L
    G_safe = jnp.maximum(G_val, 1e-32)
    return jnp.log(G_safe), dG / G_safe


def _make_fake_data(n_ch, rng):
    """Synthetic extra, spe matching the real SiPM layout (5-wide spe)."""
    spe_mean = rng.uniform(5000, 7000, size=n_ch)
    spe_sigma = rng.uniform(100, 800, size=n_ch)
    a, b = np.vectorize(reparam_from_spe)(spe_mean, spe_sigma)
    xi = rng.uniform(0.01, 0.1, size=n_ch)
    log_rho = rng.uniform(-5, -2, size=n_ch)
    log_b_ap = rng.uniform(-2, 3, size=n_ch)
    spe_all = np.stack([a, b, xi, log_rho, log_b_ap], axis=1)  # (n_ch, 5)
    ped_mean = rng.uniform(-700, -500, size=n_ch)
    ped_sigma = rng.uniform(600, 800, size=n_ch)
    extra_all = np.stack([ped_mean, ped_sigma], axis=1)  # (n_ch, 2)
    return extra_all, spe_all


def _build_pipelines(N):
    """Return (half_pipeline_fns, freq_full, dq, L) for a given N.

    Uses a symmetric q-range around 0 so build_grid produces exactly N
    grid points (span = (N-1)*dq, N = ceil(span/dq)+1 = N).
    """
    dq = 50.0
    span = (N - 1) * dq
    q_min = -span / 2.0
    q_max = span / 2.0
    N_bins = N - 1
    bins = np.linspace(q_min, q_max, N_bins + 1)
    grid = build_grid(np.zeros(N_bins), bins, A=1, q_min=q_min, q_max=q_max, dq=dq)
    assert len(grid.xsp) == N, f"grid N mismatch: {len(grid.xsp)} != {N}"

    freq_full = jnp.asarray(grid.freq)
    L = float(N * dq)

    precompute_channels_fn, precompute_event_fn, logl_and_grad_fn = make_lam_logl(
        freq_full, dq, N, _ft_extra, _ser_ft_negbin_beta_ap, _count_pgf,
    )
    return precompute_channels_fn, precompute_event_fn, logl_and_grad_fn, freq_full, dq, L


def _test_equivalence(N, rng):
    """Core: half-spectrum pipeline == full-grid reference, for a given N."""
    precompute_channels_fn, precompute_event_fn, logl_and_grad_fn, freq_full, dq, L = _build_pipelines(N)

    extra_all, spe_all = _make_fake_data(N_CH, rng)
    s_all, g0_all, u_all = precompute_channels_fn(
        jnp.asarray(extra_all), jnp.asarray(spe_all),
    )

    n_fired = 3
    Q = jnp.asarray(rng.uniform(-200, 5000, size=n_fired))
    fired_idx = jnp.asarray(rng.choice(N_CH, size=n_fired, replace=True), dtype=jnp.int32)
    u_sel, g0_phase = precompute_event_fn(Q, fired_idx, g0_all, u_all)

    lam = jnp.asarray(rng.uniform(0.1, 2.0, size=n_fired))

    logl_half, grad_half = logl_and_grad_fn(lam, u_sel, g0_phase)

    # Full-grid reference (computed independently)
    s_full = jax.vmap(lambda spe_j: _ser_ft_negbin_beta_ap(freq_full, spe_j))(jnp.asarray(spe_all))
    g0_full = jax.vmap(lambda ex_j: _ft_extra(freq_full, ex_j))(jnp.asarray(extra_all))
    u_full = jax.vmap(lambda s_j, spe_j: jax.jacfwd(
        lambda lam_s: _count_pgf(s_j, lam_s, spe_j))(0.0))(s_full, jnp.asarray(spe_all))

    g0_sel = g0_full[fired_idx]
    u_full_sel = u_full[fired_idx]
    phase = jnp.exp(1j * jnp.outer(Q.astype(jnp.float64), freq_full))
    g0_phase_full = g0_sel * phase

    logl_ref, grad_ref = _reference_full(lam, u_full_sel, g0_phase_full, L)

    assert jnp.allclose(logl_half, logl_ref, rtol=1e-5), (
        f"N={N}: logl mismatch: half={logl_half}  ref={logl_ref}"
    )
    assert jnp.allclose(grad_half, grad_ref, rtol=1e-5), (
        f"N={N}: grad mismatch: half={grad_half}  ref={grad_ref}"
    )


def test_equivalence_even_N():
    rng = np.random.default_rng(0)
    for N in [8, 12, 20]:
        _test_equivalence(N, rng)


def test_equivalence_odd_N():
    rng = np.random.default_rng(1)
    for N in [7, 11, 19]:
        _test_equivalence(N, rng)


def test_herm_w_freq_half_correctness():
    """Verify the half-spectrum frequency layout and conjugate pairing."""
    dq = 1.0
    N_even = 8
    freq_even = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(N_even, d=dq))
    N_half_even = N_even // 2 + 1  # = 5
    freq_half_even = freq_even[:N_half_even]
    assert freq_half_even[0] == 0.0  # DC
    assert abs(abs(freq_half_even[-1]) - jnp.pi / dq) < 1e-10  # Nyquist = ±π/dq for even N

    N_odd = 7
    freq_odd = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(N_odd, d=dq))
    N_half_odd = N_odd // 2 + 1  # = 4
    freq_half_odd = freq_odd[:N_half_odd]
    assert freq_half_odd[0] == 0.0  # DC
    # For odd N, last half-grid bin is NOT self-conjugate
    assert abs(freq_half_odd[-1] + freq_odd[-(N_half_odd - 1)]) < 1e-10
