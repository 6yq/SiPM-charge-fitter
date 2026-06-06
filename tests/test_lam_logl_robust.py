"""Gradient robustness: a deep-tail observed charge underflows the f32 density
G(Q), and dG/G_safe would explode (~1e20) and hijack the per-event gradient.
logl_and_grad_fn must zero the gradient where G is below the f32-resolvable
floor while keeping the NLL penalty (a finite, very negative logl).
"""

import jax
import jax.numpy as jnp
import numpy as np

from fitter.core.fft_grid import build_grid
from fitter.core.likelihood import make_lam_logl
from fitter.models.gen_tweedie import _ft_extra, _count_pgf, reparam_from_spe
from fitter.models.afterpulse import _ser_ft_negbin_beta_ap

jax.config.update("jax_enable_x64", True)


def _pipeline(dtype):
    q_min, q_max, dq = -5000.0, 200000.0, 300.0
    n_bins = int(round((q_max - q_min) / dq))
    bins = np.linspace(q_min, q_max, n_bins + 1)
    grid = build_grid(np.zeros(n_bins), bins, A=1, q_min=q_min, q_max=q_max, dq=dq)
    return make_lam_logl(jnp.asarray(grid.freq), float(grid.xsp_width),
                         len(grid.xsp), _ft_extra, _ser_ft_negbin_beta_ap,
                         _count_pgf, dtype=dtype)


def _one_channel(dtype, Q):
    pc, pe, lg = _pipeline(dtype)
    a, b = reparam_from_spe(6000.0, 435.0)
    spe = jnp.asarray([[a, b, 0.035, np.log(0.006), np.log(2.3)]])
    extra = jnp.asarray([[-600.0, 700.0]])
    _, g0_all, u_all = pc(extra, spe)
    u_sel, g0_phase = pe(jnp.asarray([Q]), jnp.asarray([0]), g0_all, u_all)
    ll, grad = lg(jnp.asarray([0.5]), u_sel, g0_phase)
    return float(ll[0]), float(grad[0])


def test_deep_tail_gradient_is_bounded_f32():
    # Q deep in the tail -> G underflows in f32. Gradient must stay finite and
    # bounded (the fix zeros it below the floor), not ~1e20.
    ll, grad = _one_channel(jnp.complex64, 71000.0)
    assert np.isfinite(ll) and np.isfinite(grad)
    assert abs(grad) < 1e6, f"deep-tail f32 gradient exploded: {grad:.3e}"
    assert ll < -20.0, f"deep-tail logl should be a strong penalty, got {ll:.2f}"


def test_normal_charge_gradient_unchanged():
    # A normal charge (well-resolved G) keeps a sane O(1-10) gradient.
    ll, grad = _one_channel(jnp.complex64, 6000.0)
    assert np.isfinite(ll) and np.isfinite(grad)
    assert abs(grad) < 1e3
