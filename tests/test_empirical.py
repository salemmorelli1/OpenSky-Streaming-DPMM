import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from opensky_streaming_dpmm.empirical import (
    BlockData,
    SnapshotData,
    analyze_results,
    execute_cell,
    normalized_mutual_information,
)
from opensky_streaming_dpmm.engine import DEVICE, DTYPE


class EmpiricalMetricTests(unittest.TestCase):
    def test_nmi_is_permutation_invariant(self):
        categories = np.asarray([0, 0, 1, 1, 2, 2])
        labels = np.asarray([8, 8, 3, 3, 5, 5])
        self.assertAlmostEqual(normalized_mutual_information(labels, categories), 1.0)

    def test_both_locked_cells_produce_finite_scores(self):
        generator = torch.Generator(device=DEVICE).manual_seed(31)
        snapshots = []
        for _ in range(2):
            x = torch.randn(18, 2, dtype=DTYPE, device=DEVICE, generator=generator)
            theta = torch.remainder(
                torch.randn(18, dtype=DTYPE, device=DEVICE, generator=generator),
                2 * math.pi,
            )
            snapshots.append(SnapshotData(x, theta, np.arange(18) % 3))
        block = BlockData("formal-0001", "formal", "a" * 64, tuple(snapshots))
        config = {
            "selected_K": 4,
            "alpha": 1.5,
            "heading_kappa": 6.0,
            "particles": 4,
            "rejuvenation_window": 4,
            "base_seed": 2026,
        }
        for method in ("streaming_vi", "collapsed_dp_smc"):
            row = execute_cell(block, method, 0.0, config)
            self.assertTrue(math.isfinite(row["mean_prequential_log_score"]))
            self.assertGreater(row["compute_latency_ms_per_observation"], 0.0)


class CompletenessGateTests(unittest.TestCase):
    @staticmethod
    def rows(blocks=2):
        result = []
        for block in range(blocks):
            for method_index, method in enumerate(("streaming_vi", "collapsed_dp_smc")):
                for dropout in (0.0, 0.15, 0.3):
                    result.append(
                        {
                            "block_id": f"formal-{block + 1:04d}",
                            "method": method,
                            "dropout_rate": dropout,
                            "mean_prequential_log_score": -5 + method_index - dropout,
                            "compute_latency_ms_per_observation": 1 + 2 * method_index + dropout,
                            "category_nmi_evaluation_only": 0.5 + 0.1 * method_index,
                            "success": True,
                        }
                    )
        return result

    def test_analysis_refuses_incomplete_factorial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.csv"
            rows = self.rows()[:-1]
            with results.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            status = root / "status.json"
            status.write_text(json.dumps({"empirical_results_available": False}))
            with self.assertRaises(ValueError):
                analyze_results(results, root / "public.json", status, expected_blocks=2)

    def test_complete_factorial_unlocks_aggregate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.csv"
            rows = self.rows()
            with results.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            status = root / "status.json"
            status.write_text(json.dumps({"empirical_results_available": False}))
            summary = analyze_results(
                results, root / "public.json", status, expected_blocks=2
            )
            self.assertEqual(summary["executions"], 12)
            self.assertTrue(json.loads(status.read_text())["empirical_results_available"])


if __name__ == "__main__":
    unittest.main()
