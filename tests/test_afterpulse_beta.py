import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from fitter.models.afterpulse import (
    NegBinBetaAPFitter,
    _beta2_charge_cf,
)
from fitter.models.gen_tweedie import reparam_from_spe


class BetaAfterpulseModelTest(unittest.TestCase):
    def test_beta2_charge_cf_is_normalized_and_has_requested_mean(self):
        charge_scale = 100.0
        mean_fraction = 0.25

        cf0 = _beta2_charge_cf(
            jnp.asarray(0.0), jnp.asarray(charge_scale), jnp.asarray(mean_fraction)
        )
        self.assertAlmostEqual(float(jnp.real(cf0)), 1.0, places=12)
        self.assertAlmostEqual(float(jnp.imag(cf0)), 0.0, places=12)

        imag_slope = jax.grad(
            lambda w: jnp.imag(
                _beta2_charge_cf(
                    jnp.asarray(w),
                    jnp.asarray(charge_scale),
                    jnp.asarray(mean_fraction),
                )
            )
        )(0.0)
        self.assertTrue(np.isfinite(float(imag_slope)))
        self.assertAlmostEqual(
            float(imag_slope), -charge_scale * mean_fraction, delta=1e-5
        )

    def test_beta_afterpulse_report_uses_single_charge_key(self):
        a, b = reparam_from_spe(120.0, 12.0)
        rho = 0.04
        beta = 0.30
        spe = jnp.array([a, b, 0.08, np.log(rho), np.log(beta / (1.0 - beta))])

        report = NegBinBetaAPFitter.spe_report(None, spe)

        self.assertNotIn("Q_ap", report)
        self.assertNotIn("Q_mean", report)
        self.assertAlmostEqual(
            report["ap_charge_mean"], beta * (1.0 - rho) * 120.0
        )
        self.assertAlmostEqual(
            report["beta_shape_b"],
            2.0 * (1.0 - beta * (1.0 - rho)) / (beta * (1.0 - rho)),
        )
        self.assertAlmostEqual(report["total_mean"], 120.0 + rho * beta * 120.0)


if __name__ == "__main__":
    unittest.main()
