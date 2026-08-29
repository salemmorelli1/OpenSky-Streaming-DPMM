import math
import unittest

import numpy as np
import torch

from opensky_streaming_dpmm.engine import (
    DEVICE,
    DTYPE,
    CollapsedDPSMC,
    OpenSkyClient,
    StreamingCircularDPMM,
    bulk_ess,
    nested_keep_mask,
    rank_normalized_split_rhat,
)


class OpenSkyContractTests(unittest.TestCase):
    def test_current_endpoint_and_bbox_keys(self):
        client = OpenSkyClient(bbox=(30.0, -125.0, 49.0, -66.0))
        self.assertEqual(client.API_URL, "https://opensky-network.org/api/states/all")
        self.assertEqual(
            set(client.params), {"lamin", "lomin", "lamax", "lomax", "extended"}
        )

    def test_invalid_bbox_is_rejected(self):
        with self.assertRaises(ValueError):
            OpenSkyClient(bbox=(50.0, 10.0, 40.0, 20.0))


class ExperimentalDesignTests(unittest.TestCase):
    def test_nested_masks_are_reproducible_and_nested(self):
        full = nested_keep_mask(1000, 0.0, 2026)
        keep_15 = nested_keep_mask(1000, 0.15, 2026)
        keep_30 = nested_keep_mask(1000, 0.30, 2026)
        self.assertTrue(np.all(keep_30 <= keep_15))
        self.assertTrue(np.all(keep_15 <= full))
        np.testing.assert_array_equal(keep_15, nested_keep_mask(1000, 0.15, 2026))


class VariationalInvariantTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device=DEVICE).manual_seed(100)
        self.x = torch.randn(64, 2, dtype=DTYPE, device=DEVICE, generator=generator)
        self.theta = torch.remainder(
            torch.randn(64, dtype=DTYPE, device=DEVICE, generator=generator),
            2.0 * math.pi,
        )

    def test_residual_stick_normalizes(self):
        model = StreamingCircularDPMM(max_clusters=8)
        weights = model.expected_weights()
        self.assertEqual(weights.shape[0], 8)
        self.assertTrue(torch.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)

    def test_responsibilities_normalize(self):
        model = StreamingCircularDPMM(max_clusters=8)
        responsibilities = model.partial_fit(self.x, self.theta)
        expected = torch.ones(self.x.shape[0], dtype=DTYPE, device=DEVICE)
        self.assertTrue(torch.allclose(responsibilities.sum(1), expected, atol=1e-9))
        self.assertTrue(torch.isfinite(responsibilities).all())


class SMCAndDiagnosticTests(unittest.TestCase):
    def test_particle_filter_smoke(self):
        generator = torch.Generator(device=DEVICE).manual_seed(101)
        x = torch.randn(30, 2, dtype=DTYPE, device=DEVICE, generator=generator)
        theta = torch.remainder(
            torch.randn(30, dtype=DTYPE, device=DEVICE, generator=generator),
            2.0 * math.pi,
        )
        model = CollapsedDPSMC(x[:10], particles=16, rejuvenation_window=5)
        ess = model.partial_fit(x, theta)
        self.assertTrue(math.isfinite(ess))
        self.assertGreaterEqual(ess, 1.0)
        self.assertLessEqual(ess, 16.0)

    def test_rank_diagnostics_distinguish_shifted_chains(self):
        rng = np.random.default_rng(2026)
        converged = rng.normal(size=(4, 500))
        shifted = converged.copy()
        shifted[-1] += 4.0
        converged_rhat = rank_normalized_split_rhat(converged)["max"]
        shifted_rhat = rank_normalized_split_rhat(shifted)["max"]
        self.assertLess(converged_rhat, shifted_rhat)
        self.assertTrue(math.isfinite(bulk_ess(converged)))


if __name__ == "__main__":
    unittest.main()
