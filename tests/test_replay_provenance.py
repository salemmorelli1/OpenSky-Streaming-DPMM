import copy
import json
from unittest.mock import patch

import pytest

from opensky_streaming_dpmm.empirical import (
    analyze_results,
    calibrate_truncation,
    load_locked_config,
    run_factorial,
)
from test_empirical import (
    analysis_fixture,
    digest,
    write_blocks,
    write_csv,
    write_lock,
)


def rehash_lock(path, lock):
    unsigned = dict(lock)
    unsigned.pop("lock_sha256", None)
    lock["lock_sha256"] = digest(unsigned)
    path.write_text(json.dumps(lock))


@pytest.mark.parametrize(
    "field,value",
    [
        ("mean_prequential_log_score", -100),
        ("retained_observations", 999),
        ("scored_observations", 0),
        ("category_observations", 0),
        ("occupied_clusters", 2.5),
        ("compute_latency_ms", 100000),
        ("endpoint_protocol", "old-sequential-score"),
        ("implementation_sha256", "f" * 64),
        ("execution_environment_sha256", "f" * 64),
        ("block_sha256", "f" * 64),
        ("success", False),
    ],
)
def test_modified_finite_checkpoint_cannot_resume(tmp_path, field, value):
    lock_path, _, paths, rows, results, _ = analysis_fixture(tmp_path)
    rows[0][field] = value
    write_csv(results, rows)
    before = results.read_bytes()
    with patch("opensky_streaming_dpmm.empirical.execute_cell") as execute:
        with pytest.raises(ValueError):
            run_factorial(paths, lock_path, results, 2, validation_only=True)
    execute.assert_not_called()
    assert results.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.append(copy.deepcopy(rows[0])),
        lambda rows: rows[0].update(method="unknown"),
        lambda rows: rows[0].update(dropout_rate=0.2),
    ],
)
def test_duplicate_or_out_of_design_checkpoints_fail_closed(tmp_path, mutation):
    lock_path, _, paths, rows, results, _ = analysis_fixture(tmp_path)
    mutation(rows)
    write_csv(results, rows)
    with pytest.raises(ValueError):
        run_factorial(paths, lock_path, results, 2, validation_only=True)


def test_formal_windows_must_follow_calibration(tmp_path):
    lock_path, _ = write_lock(tmp_path)
    paths = write_blocks(tmp_path, start=500)
    with pytest.raises(ValueError, match="precede calibration"):
        run_factorial(
            paths, lock_path, tmp_path / "results.csv", 2, validation_only=True
        )
    assert not (tmp_path / "results.csv").exists()


def test_different_box_cannot_enter_same_lock(tmp_path):
    lock_path, lock = write_lock(tmp_path)
    contract = dict(lock["collection_contract"], bbox=[30, -125, 49, -66])
    paths = write_blocks(tmp_path, contract=contract)
    with pytest.raises(ValueError, match="different acquisition protocol"):
        run_factorial(
            paths, lock_path, tmp_path / "results.csv", 2, validation_only=True
        )


def test_changed_compute_environment_cannot_resume(tmp_path):
    lock_path, _, paths, _, results, _ = analysis_fixture(tmp_path)
    with patch(
        "opensky_streaming_dpmm.empirical.execution_environment",
        return_value={"device": "another machine"},
    ):
        with pytest.raises(ValueError, match="compute environment changed"):
            run_factorial(paths, lock_path, results, 2, validation_only=True)


def test_rehashed_lock_for_another_implementation_is_rejected(tmp_path):
    path, lock = write_lock(tmp_path)
    lock["implementation_sha256"] = "f" * 64
    rehash_lock(path, lock)
    with pytest.raises(ValueError, match="another implementation"):
        load_locked_config(path)


def test_old_schema_cannot_be_rehashed_into_current_evidence(tmp_path):
    path, lock = write_lock(tmp_path)
    lock["schema"] = "opensky-frozen-analysis-v1"
    rehash_lock(path, lock)
    with pytest.raises(ValueError, match="invalid or unlocked"):
        load_locked_config(path)


def test_analysis_requires_original_formal_blocks(tmp_path):
    lock_path, _, _, _, results, status = analysis_fixture(tmp_path)
    with pytest.raises(ValueError, match="original verified formal blocks"):
        analyze_results(
            results,
            lock_path,
            tmp_path / "summary.json",
            status,
            2,
            validation_only=True,
        )


def test_analysis_rejects_changed_formal_block_even_when_csv_is_complete(tmp_path):
    lock_path, _, paths, _, results, status = analysis_fixture(tmp_path)
    # Replace a verified block with a different source window, preserving its id.
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    new_paths = write_blocks(replacement, count=1, start=1500)
    paths[0].write_bytes(new_paths[0].read_bytes())
    with pytest.raises(ValueError, match="block hash changed"):
        analyze_results(
            results,
            lock_path,
            tmp_path / "summary.json",
            status,
            2,
            block_paths=paths,
            validation_only=True,
        )
    assert not json.loads(status.read_text())["empirical_results_available"]
    assert not (tmp_path / "summary.json").exists()


def test_public_gate_cannot_be_lowered_by_expected_blocks_argument(tmp_path):
    lock_path, _, paths, _, results, status = analysis_fixture(tmp_path)
    with pytest.raises(ValueError, match="100 blocks and 600 cells"):
        analyze_results(
            results, lock_path, tmp_path / "summary.json", status, 2, block_paths=paths
        )


def test_real_gzip_calibration_factorial_resume_and_analysis_stay_private(tmp_path):
    calibration = tmp_path / "calibration"
    formal = tmp_path / "formal"
    calibration.mkdir()
    formal.mkdir()
    calibration_paths = write_blocks(
        calibration, count=2, phase="calibration", start=100
    )
    lock_path = tmp_path / "real-lock.json"
    lock = calibrate_truncation(
        calibration_paths,
        lock_path,
        candidate_k=(4, 8, 12),
        alpha=0.1,
        particles=4,
        expected_blocks=2,
        tail_threshold=0.01,
        saturation_threshold=0.05,
        validation_only=True,
    )
    assert load_locked_config(lock_path)["lock_sha256"] == lock["lock_sha256"]
    assert lock["scope"] == "validation_only"
    assert all(
        item["tail_weight_q95"] is None or item["tail_weight_q95"] >= 0
        for item in lock["candidate_diagnostics"]
    )
    formal_paths = write_blocks(formal, count=2, start=2000)
    results = tmp_path / "actual-results.csv"
    first = run_factorial(formal_paths, lock_path, results, 2, validation_only=True)
    assert len(first) == 12
    before = results.read_bytes()
    with patch("opensky_streaming_dpmm.empirical.execute_cell") as execute:
        resumed = run_factorial(
            formal_paths, lock_path, results, 2, validation_only=True
        )
    execute.assert_not_called()
    assert len(resumed) == 12 and before == results.read_bytes()
    status = tmp_path / "status.json"
    status.write_text('{"empirical_results_available": false}')
    summary = analyze_results(
        results,
        lock_path,
        tmp_path / "private-summary.json",
        status,
        2,
        block_paths=formal_paths,
        validation_only=True,
    )
    assert not summary["empirical_results_available"]
    assert not json.loads(status.read_text())["empirical_results_available"]
    assert "NaN" not in (tmp_path / "private-summary.json").read_text()
    with pytest.raises(ValueError, match="overwrite an existing lock"):
        calibrate_truncation(
            calibration_paths,
            lock_path,
            candidate_k=(4, 8, 12),
            alpha=0.1,
            particles=4,
            expected_blocks=2,
            tail_threshold=0.01,
            saturation_threshold=0.05,
            validation_only=True,
        )


@pytest.mark.parametrize(
    "overrides", [dict(particles=4), dict(alpha=0.1), dict(selected_K=32)]
)
def test_confirmatory_lock_cannot_change_frozen_design_settings(tmp_path, overrides):
    path, _ = write_lock(tmp_path, scope="confirmatory", **overrides)
    with pytest.raises(ValueError):
        load_locked_config(path)
