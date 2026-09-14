import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from opensky_streaming_dpmm.empirical import (
    BlockData,
    DROPOUT_LEVELS,
    LOCK_SCHEMA,
    PRIMARY_ENDPOINTS,
    SnapshotData,
    analyze_results,
    calibrate_truncation,
    execute_cell,
    normalized_mutual_information,
    run_factorial,
)
from opensky_streaming_dpmm.engine import DEVICE, DTYPE


class EmpiricalMetricTests(unittest.TestCase):
    def test_nmi_is_permutation_invariant(self):
        categories = np.asarray([0, 0, 1, 1, 2, 2])
        labels = np.asarray([8, 8, 3, 3, 5, 5])
        self.assertAlmostEqual(normalized_mutual_information(labels, categories), 1.0)

    def test_nmi_ignores_opensky_no_information_categories(self):
        categories = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
        labels = np.asarray([0, 1, 0, 1, 7, 7, 8, 8])
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

    def test_vi_accepts_smaller_later_streaming_batch(self):
        generator = torch.Generator(device=DEVICE).manual_seed(32)
        sizes = (8, 3)
        snapshots = []
        for size in sizes:
            x = torch.randn(size, 2, dtype=DTYPE, device=DEVICE, generator=generator)
            theta = torch.remainder(
                torch.randn(size, dtype=DTYPE, device=DEVICE, generator=generator),
                2 * math.pi,
            )
            snapshots.append(SnapshotData(x, theta, np.full(size, 2)))
        block = BlockData("formal-0001", "formal", "a" * 64, tuple(snapshots))
        config = {
            "selected_K": 4,
            "alpha": 1.5,
            "heading_kappa": 6.0,
            "base_seed": 2026,
        }
        row = execute_cell(block, "streaming_vi", 0.0, config)
        self.assertTrue(math.isfinite(row["mean_prequential_log_score"]))


def write_lock(root: Path) -> tuple[Path, dict]:
    lock = {
        "schema": LOCK_SCHEMA,
        "status": "locked",
        "calibration_block_ids": [f"calibration-{index:04d}" for index in range(1, 11)],
        "calibration_block_sha256": [f"{index:064x}" for index in range(1, 11)],
        "selected_K": 4,
        "alpha": 1.5,
        "heading_kappa": 6.0,
        "particles": 4,
        "rejuvenation_window": 4,
        "dropout_levels": list(DROPOUT_LEVELS),
        "base_seed": 2026,
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "secondary_endpoint": "category_nmi_evaluation_only",
    }
    encoded = json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    lock["lock_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    path = root / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path, lock


class CompletenessGateTests(unittest.TestCase):
    @staticmethod
    def rows(lock_sha: str, blocks=2):
        result = []
        for block in range(blocks):
            for method_index, method in enumerate(("streaming_vi", "collapsed_dp_smc")):
                for dropout in (0.0, 0.15, 0.3):
                    result.append(
                        {
                            "block_id": f"formal-{block + 1:04d}",
                            "block_sha256": hashlib.sha256(
                                f"formal-{block + 1:04d}".encode()
                            ).hexdigest(),
                            "lock_sha256": lock_sha,
                            "method": method,
                            "dropout_rate": dropout,
                            "mean_prequential_log_score": -5 + method_index - dropout,
                            "compute_latency_ms_per_observation": 1 + 2 * method_index + dropout,
                            "occupied_clusters": 3 + method_index,
                            "category_nmi_evaluation_only": 0.5 + 0.1 * method_index,
                            "success": True,
                        }
                    )
        return result

    def test_analysis_refuses_incomplete_factorial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, lock = write_lock(root)
            results = root / "results.csv"
            rows = self.rows(lock["lock_sha256"])[:-1]
            with results.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            status = root / "status.json"
            status.write_text(json.dumps({"empirical_results_available": False}))
            with self.assertRaises(ValueError):
                analyze_results(
                    results, lock_path, root / "public.json", status, expected_blocks=2
                )

    def test_complete_factorial_unlocks_aggregate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, lock = write_lock(root)
            results = root / "results.csv"
            rows = self.rows(lock["lock_sha256"])
            with results.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            status = root / "status.json"
            status.write_text(json.dumps({"empirical_results_available": False}))
            summary = analyze_results(
                results, lock_path, root / "public.json", status, expected_blocks=2
            )
            self.assertEqual(summary["executions"], 12)
            self.assertIn("occupied_clusters", summary["analysis"])
            contrast = summary["analysis"]["occupied_clusters"][
                "paired_architecture_by_dropout"
            ]["0.0"]
            self.assertIn("p_value_holm", contrast)
            self.assertEqual(summary["provenance"]["analysis_lock_sha256"], lock["lock_sha256"])
            self.assertTrue(json.loads(status.read_text())["empirical_results_available"])

    def test_analysis_rejects_tampered_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, lock = write_lock(root)
            lock["alpha"] = 999
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            results = root / "results.csv"
            rows = self.rows(lock["lock_sha256"])
            with results.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            status = root / "status.json"
            status.write_text(json.dumps({"empirical_results_available": False}))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                analyze_results(
                    results, lock_path, root / "public.json", status, expected_blocks=2
                )

    def test_analysis_rejects_nonfinite_primary_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, lock = write_lock(root)
            results = root / "results.csv"
            rows = self.rows(lock["lock_sha256"])
            rows[0]["mean_prequential_log_score"] = float("nan")
            with results.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            status = root / "status.json"
            status.write_text(json.dumps({"empirical_results_available": False}))
            with self.assertRaisesRegex(ValueError, "non-finite primary endpoint"):
                analyze_results(
                    results, lock_path, root / "public.json", status, expected_blocks=2
                )


class FormalIntegrityTests(unittest.TestCase):
    def test_calibration_requires_ten_unique_blocks(self):
        block = BlockData("calibration-0001", "calibration", "a" * 64, tuple())
        with tempfile.TemporaryDirectory() as directory, patch(
            "opensky_streaming_dpmm.empirical.load_block", return_value=block
        ):
            with self.assertRaisesRegex(ValueError, "exactly 10"):
                calibrate_truncation([Path("one.json.gz")], Path(directory) / "lock.json")

    def test_resume_rejects_checkpoint_from_another_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, _ = write_lock(root)
            output = root / "results.csv"
            rows = CompletenessGateTests.rows("f" * 64, blocks=1)
            with output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            block = BlockData(
                "formal-0001",
                "formal",
                rows[0]["block_sha256"],
                tuple(),
            )
            with patch(
                "opensky_streaming_dpmm.empirical.load_block", return_value=block
            ):
                with self.assertRaisesRegex(ValueError, "verified analysis lock"):
                    run_factorial(
                        [Path("formal-0001.json.gz")],
                        lock_path,
                        output,
                        expected_blocks=1,
                    )


if __name__ == "__main__":
    unittest.main()
