import csv
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from opensky_streaming_dpmm.collector import (
    CollectionConfig,
    atomic_write_gzip_json,
    build_block,
)
from opensky_streaming_dpmm.contracts import ENDPOINT_PROTOCOL
from opensky_streaming_dpmm.empirical import (
    AMENDED_ACQUISITION,
    BlockData,
    DROPOUT_LEVELS,
    LOCK_SCHEMA,
    METHODS,
    PRIMARY_ENDPOINTS,
    SnapshotData,
    analyze_results,
    calibrate_truncation,
    execute_cell,
    execution_environment,
    implementation_sha256,
    load_block,
    load_locked_config,
    normalized_mutual_information,
    result_sha256,
    run_factorial,
    thinned_snapshots,
)
from opensky_streaming_dpmm.engine import DEVICE, DTYPE


VALIDATION_ACQUISITION = {
    "bbox": [49, 7, 54, 13],
    "snapshots_per_block": 2,
    "interval_seconds": 0.0,
    "minimum_eligible_states": 12,
    "max_attempts_per_block": 4,
    "retry_delay_seconds": 0.0,
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def write_lock(root: Path, scope="validation_only", **overrides):
    runtime = execution_environment()
    lock = {
        "schema": LOCK_SCHEMA,
        "status": "locked",
        "scope": scope,
        "endpoint_protocol": ENDPOINT_PROTOCOL,
        "implementation_sha256": implementation_sha256(),
        "collection_contract": AMENDED_ACQUISITION
        if scope == "confirmatory"
        else VALIDATION_ACQUISITION,
        "calibration_api_time_end": 1000,
        "calibration_block_ids": [f"calibration-{i:04d}" for i in range(1, 11)],
        "calibration_block_sha256": [f"{i:064x}" for i in range(1, 11)],
        "selected_K": 4,
        "alpha": 1.5,
        "heading_kappa": 6.0,
        "particles": 4,
        "base_mean": [4.5, 0.0],
        "base_variance": [0.4, 1.2],
        "rejuvenation_window": 4,
        "dropout_levels": list(DROPOUT_LEVELS),
        "base_seed": 2026,
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "secondary_endpoint": "category_nmi_evaluation_only",
        "execution_environment": runtime,
        "execution_environment_sha256": digest(runtime),
    }
    if scope == "confirmatory":
        lock.update(
            selected_K=8,
            particles=64,
            rejuvenation_window=24,
            tail_threshold=0.01,
            saturation_threshold=0.05,
            candidate_diagnostics=[
                {
                    "K": k,
                    "usable": k == 8,
                    "tail_weight_q95": 0.005 if k == 8 else None,
                    "saturation_rate": 0.0 if k == 8 else None,
                    "passed": k == 8,
                }
                for k in (8, 12, 16, 24, 32)
            ],
        )
    lock.update(overrides)
    lock["lock_sha256"] = digest(lock)
    path = root / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path, lock


class SyntheticClient:
    def __init__(self, count, start):
        self.count, self.time = count, start

    def get_states(self, bbox):
        self.time += 5
        rows = []
        for i in range(self.count):
            row = [None] * 18
            row[0] = f"{i:06x}"
            row[3], row[4] = self.time - 1, self.time
            row[5], row[6], row[8] = 8.2, 50.1, False
            row[9] = 80.0 + 90.0 * (i % 3) + (i % 5)
            row[10] = 120.0 * (i % 3) + (i % 7)
            row[11] = (i % 3 - 1) * 4.0 + 0.1 * (i % 5)
            row[17] = 2 + i % 3
            rows.append(row)
        return {"time": self.time, "states": rows}, {"network_latency_ms": 1.0}


def write_blocks(root, count=2, contract=None, phase="formal", start=2000):
    contract = contract or VALIDATION_ACQUISITION
    config = CollectionConfig(phase=phase, **contract)
    paths = []
    for i in range(count):
        name = f"{phase}-{i + 1:04d}"
        client = SyntheticClient(
            max(18, config.minimum_eligible_states), start + i * 100
        )
        with patch("opensky_streaming_dpmm.collector.time.sleep"):
            block = build_block(client, config, b"x" * 32, name)
        path = root / f"{name}.json.gz"
        atomic_write_gzip_json(path, block)
        paths.append(path)
    return paths


def rows_for(lock, paths):
    rows = []
    for path in paths:
        block = load_block(path)
        for m, method in enumerate(METHODS):
            for dropout in DROPOUT_LEVELS:
                snapshots = thinned_snapshots(block, dropout, lock["base_seed"])
                retained = sum(s.x.shape[0] for s in snapshots)
                category_count = sum(
                    np.count_nonzero((s.category >= 2) & (s.category != 13))
                    for s in snapshots
                )
                per_observation = 1 + 2 * m + dropout
                row = {
                    "block_id": block.block_id,
                    "block_sha256": block.content_sha256,
                    "lock_sha256": lock["lock_sha256"],
                    "method": method,
                    "dropout_rate": dropout,
                    "retained_observations": retained,
                    "scored_observations": retained - snapshots[0].x.shape[0],
                    "category_observations": int(category_count),
                    "mean_prequential_log_score": -5 + m - dropout,
                    "compute_latency_ms": per_observation * retained,
                    "compute_latency_ms_per_observation": per_observation,
                    "occupied_clusters": 3 + m,
                    "category_nmi_evaluation_only": 0.5 + 0.1 * m,
                    "tail_weight": 0.005 if m == 0 else float("nan"),
                    "particle_ess": float("nan") if m == 0 else 3.0,
                    "device": str(DEVICE),
                    "endpoint_protocol": ENDPOINT_PROTOCOL,
                    "implementation_sha256": lock["implementation_sha256"],
                    "execution_environment_sha256": lock[
                        "execution_environment_sha256"
                    ],
                    "success": True,
                }
                row["result_sha256"] = result_sha256(row)
                rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analysis_fixture(root, count=2, scope="validation_only"):
    lock_path, lock = write_lock(root, scope)
    paths = write_blocks(root, count, lock["collection_contract"])
    rows = rows_for(lock, paths)
    results = root / "results.csv"
    write_csv(results, rows)
    status = root / "status.json"
    status.write_text(json.dumps({"empirical_results_available": False}))
    return lock_path, lock, paths, rows, results, status


def snapshots(sizes=(18, 18)):
    generator = torch.Generator(device=DEVICE).manual_seed(31)
    return tuple(
        SnapshotData(
            torch.randn(n, 2, dtype=DTYPE, device=DEVICE, generator=generator),
            torch.remainder(
                torch.randn(n, dtype=DTYPE, device=DEVICE, generator=generator),
                2 * math.pi,
            ),
            2 + np.arange(n) % 3,
        )
        for n in sizes
    )


def test_nmi_is_permutation_invariant():
    categories = np.asarray([2, 2, 3, 3, 4, 4])
    assert normalized_mutual_information(
        np.asarray([8, 8, 3, 3, 5, 5]), categories
    ) == pytest.approx(1)


def test_nmi_ignores_no_information_and_reserved_categories():
    categories = np.asarray([0, 0, 1, 1, 13, 13, 2, 2, 3, 3])
    labels = np.asarray([0, 1, 0, 1, 0, 1, 7, 7, 8, 8])
    assert normalized_mutual_information(labels, categories) == pytest.approx(1)


@pytest.mark.parametrize("method", METHODS)
def test_both_cells_produce_finite_scores(method):
    block = BlockData("formal-0001", "formal", "a" * 64, snapshots())
    config = dict(
        selected_K=4,
        alpha=1.5,
        heading_kappa=6.0,
        particles=4,
        rejuvenation_window=4,
        base_seed=2026,
    )
    row = execute_cell(block, method, 0.0, config)
    assert math.isfinite(row["mean_prequential_log_score"])
    assert row["compute_latency_ms_per_observation"] > 0
    assert row["scored_observations"] == 18


def test_vi_accepts_smaller_later_batch():
    block = BlockData("formal-0001", "formal", "a" * 64, snapshots((8, 3)))
    config = dict(selected_K=4, alpha=1.5, heading_kappa=6.0, base_seed=2026)
    assert math.isfinite(
        execute_cell(block, "streaming_vi", 0.0, config)["mean_prequential_log_score"]
    )


def test_analysis_refuses_incomplete_factorial(tmp_path):
    lock_path, _, paths, rows, results, status = analysis_fixture(tmp_path)
    write_csv(results, rows[:-1])
    with pytest.raises(ValueError, match="12 unique cells"):
        analyze_results(
            results,
            lock_path,
            tmp_path / "public.json",
            status,
            2,
            block_paths=paths,
            validation_only=True,
        )
    assert not (tmp_path / "public.json").exists()


def test_small_validation_cannot_unlock_publication(tmp_path):
    lock_path, lock, paths, _, results, status = analysis_fixture(tmp_path)
    before = status.read_bytes()
    summary = analyze_results(
        results,
        lock_path,
        tmp_path / "validation.json",
        status,
        2,
        block_paths=paths,
        validation_only=True,
    )
    assert summary["executions"] == 12
    assert not summary["empirical_results_available"]
    assert status.read_bytes() == before
    contrast = summary["analysis"]["occupied_clusters"][
        "paired_architecture_by_dropout"
    ]["0.0"]
    assert contrast["estimate"] == pytest.approx(1)
    assert contrast["p_value_holm"] == 0
    assert summary["provenance"]["analysis_lock_sha256"] == lock["lock_sha256"]


def test_only_full_confirmatory_design_unlocks_aggregate(tmp_path):
    lock_path, _, paths, _, results, status = analysis_fixture(
        tmp_path, 100, "confirmatory"
    )
    summary = analyze_results(
        results, lock_path, tmp_path / "aggregate.json", status, block_paths=paths
    )
    assert summary["blocks"] == 100 and summary["executions"] == 600
    assert summary["empirical_results_available"]
    assert json.loads(status.read_text())["empirical_results_available"]
    assert summary["hardware"] == execution_environment()
    assert "track_key" not in (tmp_path / "aggregate.json").read_text()


def test_analysis_rejects_tampered_lock(tmp_path):
    lock_path, lock, paths, _, results, status = analysis_fixture(tmp_path)
    lock["alpha"] = 999
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        analyze_results(
            results,
            lock_path,
            tmp_path / "public.json",
            status,
            2,
            block_paths=paths,
            validation_only=True,
        )


def test_analysis_rejects_nonfinite_primary_endpoint(tmp_path):
    lock_path, _, paths, rows, results, status = analysis_fixture(tmp_path)
    rows[0]["mean_prequential_log_score"] = float("nan")
    write_csv(results, rows)
    with pytest.raises(ValueError, match="non-finite primary endpoint"):
        analyze_results(
            results,
            lock_path,
            tmp_path / "public.json",
            status,
            2,
            block_paths=paths,
            validation_only=True,
        )


def test_calibration_requires_ten_blocks(tmp_path):
    with pytest.raises(ValueError, match="exactly 10"):
        calibrate_truncation([], tmp_path / "lock.json")


def test_resume_rejects_checkpoint_from_another_lock(tmp_path):
    lock_path, _, paths, rows, results, _ = analysis_fixture(tmp_path)
    for row in rows:
        row["lock_sha256"] = "f" * 64
    write_csv(results, rows)
    with pytest.raises(ValueError, match="verified analysis lock"):
        run_factorial(paths, lock_path, results, 2, validation_only=True)


def test_rehashed_invalid_hyperparameter_lock_is_rejected(tmp_path):
    path, _ = write_lock(tmp_path, alpha=-1)
    with pytest.raises(ValueError, match="alpha"):
        load_locked_config(path)
