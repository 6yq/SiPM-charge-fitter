import unittest

import numpy as np

from fitter.models.afterpulse import NegBinBetaAPFitter
from fitter.models.gen_tweedie import spe_from_reparam
from scripts.fit import fit_optax, make_fitter, theta_from_initial_values


class FitBetaApiTest(unittest.TestCase):
    def test_make_fitter_uses_beta_model_and_lam_init(self):
        charges = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
        counts = np.array([2.0, 5.0, 8.0, 3.0, 1.0])

        fitter = make_fitter(
            charges,
            counts,
            ped_mean=0.0,
            ped_sigma=3.0,
            spe_mean=20.0,
            spe_sigma=4.0,
            lam_init=0.37,
        )

        self.assertIsInstance(fitter, NegBinBetaAPFitter)
        self.assertAlmostEqual(float(fitter.init[fitter.layout["lam"].start]), 0.37)

    def test_theta_from_initial_values_accepts_physical_fit_result_shape(self):
        charges = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
        counts = np.array([2.0, 5.0, 8.0, 3.0, 1.0])
        fitter = make_fitter(
            charges,
            counts,
            ped_mean=0.0,
            ped_sigma=3.0,
            spe_mean=20.0,
            spe_sigma=4.0,
            lam_init=0.2,
        )

        theta = theta_from_initial_values(
            fitter,
            {
                "ped": {"ped_mean": 1.0, "ped_sigma": 2.5},
                "spe": {
                    "spe_mean": 22.0,
                    "spe_sigma": 3.5,
                    "xi": 0.06,
                    "rho": 0.03,
                    "beta": 0.22,
                },
                "lam": 0.71,
            },
        )

        extra = theta[fitter.layout["extra"]]
        spe = theta[fitter.layout["spe"]]
        self.assertAlmostEqual(float(extra[0]), 1.0)
        self.assertAlmostEqual(float(extra[1]), 2.5)
        mean, sigma = spe_from_reparam(spe[0], spe[1])
        self.assertAlmostEqual(float(mean), 22.0)
        self.assertAlmostEqual(float(sigma), 3.5)
        self.assertAlmostEqual(float(spe[2]), 0.06)
        self.assertAlmostEqual(float(np.exp(spe[3])), 0.03)
        self.assertAlmostEqual(float(1.0 / (1.0 + np.exp(-spe[4]))), 0.22)
        self.assertAlmostEqual(float(theta[fitter.layout["lam"].start]), 0.71)

    def test_fixed_step_lbfgs_runs_without_linesearch_state(self):
        charges = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
        counts = np.array([2.0, 5.0, 8.0, 3.0, 1.0])
        fitter = make_fitter(
            charges,
            counts,
            ped_mean=0.0,
            ped_sigma=3.0,
            spe_mean=20.0,
            spe_sigma=4.0,
            lam_init=0.2,
        )

        theta, logl, _converged, n_iter, trace_logl, trace_gnorm = fit_optax(
            fitter, maxiter=1, optimizer_name="fixed"
        )

        self.assertEqual(n_iter, 1)
        self.assertEqual(len(theta), len(fitter.init))
        self.assertTrue(np.isfinite(logl))
        self.assertTrue(np.all(np.isfinite(trace_logl)))
        self.assertTrue(np.all(np.isfinite(trace_gnorm)))


if __name__ == "__main__":
    unittest.main()
