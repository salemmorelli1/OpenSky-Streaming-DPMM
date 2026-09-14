"""Locked calibration, factorial replay, and paired analysis for OpenSky blocks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import t as student_t

from .collector import read_verified_block
from .engine import (
    DEVICE,
    DTYPE,
    CollapsedDPSMC,
    StreamingCircularDPMM,
    nested_keep_mask,
    synchronized_latency_ms,
)


METHODS = ("streaming_vi", "collapsed_dp_smc")
DROPOUT_LEVELS = (0.0, 0.15, 0.30)
LOCK_SCHEMA = "opensky-frozen-analysis-v1"
PRIMARY_ENDPOINTS = (
    "mean_prequential_log_score",
    "compute_latency_ms_per_observation",
    "occupied_clusters",
)


@dataclass(frozen=True)
class SnapshotData:
    x: torch.Tensor
    theta: torch.Tensor
    category: np.ndarray


@dataclass(frozen=True)
class BlockData:
    block_id: str
    phase: str
    content_sha256: str
    snapshots: tuple[SnapshotData, ...]


def _stable_seed(*parts: object) -> int:
    value = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_locked_config(path: Path) -> dict[str, Any]:
    """Load a lock only when its embedded fingerprint matches its contents."""

    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != LOCK_SCHEMA or config.get("status") != "locked":
        raise ValueError("invalid or unlocked analysis configuration")
    expected = config.get("lock_sha256")
    fingerprint_input = dict(config)
    fingerprint_input.pop("lock_sha256", None)
    actual = _canonical_sha256(fingerprint_input)
    if not isinstance(expected, str) or not hmac.compare_digest(expected, actual):
        raise ValueError("analysis lock fingerprint mismatch")
    if tuple(config.get("dropout_levels", ())) != DROPOUT_LEVELS:
        raise ValueError("analysis lock does not match the implemented factorial design")
    if tuple(config.get("primary_endpoints", ())) != PRIMARY_ENDPOINTS:
        raise ValueError("analysis lock does not match the primary endpoints")
    return config


def load_block(path: Path) -> BlockData:
    payload = read_verified_block(path)
    snapshots: list[SnapshotData] = []
    for snapshot in payload["snapshots"]:
        observations = snapshot["observations"]
        x = torch.as_tensor(
            [
                [math.log1p(float(row["velocity"])), math.asinh(float(row["vertical_rate"]) / 5.0)]
                for row in observations
            ],
            dtype=DTYPE,
            device=DEVICE,
        ).reshape(-1, 2)
        theta = torch.as_tensor(
            [math.radians(float(row["true_track"]) % 360.0) for row in observations],
            dtype=DTYPE,
            device=DEVICE,
        )
        category = np.asarray(
            [-1 if row.get("category") is None else int(row["category"]) for row in observations],
            dtype=np.int64,
        )
        snapshots.append(SnapshotData(x=x, theta=theta, category=category))
    return BlockData(
        block_id=str(payload["block_id"]),
        phase=str(payload["phase"]),
        content_sha256=str(payload["integrity"]["content_sha256"]),
        snapshots=tuple(snapshots),
    )


def normalized_mutual_information(labels: np.ndarray, categories: np.ndarray) -> float:
    """Arithmetic-mean normalized mutual information, invariant to label names."""

    labels = np.asarray(labels)
    categories = np.asarray(categories)
    # OpenSky category 0 means "no information" and 1 means "no ADS-B
    # category information"; neither is an external class label.
    valid = categories >= 2
    labels, categories = labels[valid], categories[valid]
    if labels.size < 2:
        return float("nan")
    _, row = np.unique(labels, return_inverse=True)
    _, column = np.unique(categories, return_inverse=True)
    contingency = np.zeros((row.max() + 1, column.max() + 1), dtype=float)
    np.add.at(contingency, (row, column), 1.0)
    probability = contingency / contingency.sum()
    row_probability = probability.sum(axis=1)
    column_probability = probability.sum(axis=0)
    nz = probability > 0
    denominator = row_probability[:, None] * column_probability[None, :]
    mutual_information = float(
        np.sum(probability[nz] * np.log(probability[nz] / denominator[nz]))
    )
    h_row = float(-np.sum(row_probability[row_probability > 0] * np.log(row_probability[row_probability > 0])))
    h_column = float(-np.sum(column_probability[column_probability > 0] * np.log(column_probability[column_probability > 0])))
    entropy_mean = 0.5 * (h_row + h_column)
    return 1.0 if entropy_mean == 0 and np.array_equal(row, column) else (
        mutual_information / entropy_mean if entropy_mean > 0 else 0.0
    )


def thinned_snapshots(
    block: BlockData, dropout_rate: float, base_seed: int
) -> tuple[SnapshotData, ...]:
    result: list[SnapshotData] = []
    for index, snapshot in enumerate(block.snapshots):
        seed = _stable_seed(base_seed, block.block_id, index)
        mask = nested_keep_mask(snapshot.x.shape[0], dropout_rate, seed)
        selected = np.flatnonzero(mask)
        tensor_index = torch.as_tensor(selected, dtype=torch.long, device=DEVICE)
        result.append(
            SnapshotData(
                x=snapshot.x[tensor_index],
                theta=snapshot.theta[tensor_index],
                category=snapshot.category[selected],
            )
        )
    return tuple(result)


def calibrate_truncation(
    block_paths: Iterable[Path],
    output: Path,
    candidate_k: tuple[int, ...] = (8, 12, 16, 24, 32),
    alpha: float = 1.5,
    heading_kappa: float = 6.0,
    tail_threshold: float = 0.01,
    saturation_threshold: float = 0.05,
    particles: int = 64,
    seed: int = 2026,
    expected_blocks: int = 10,
) -> dict[str, Any]:
    """Select K using calibration blocks that are excluded from formal analysis."""

    blocks = [load_block(path) for path in sorted(block_paths)]
    if len(blocks) != expected_blocks:
        raise ValueError(f"calibration requires exactly {expected_blocks} verified blocks")
    if any(block.phase != "calibration" for block in blocks):
        raise ValueError("calibration requires calibration-phase blocks")
    if len({block.block_id for block in blocks}) != expected_blocks:
        raise ValueError("duplicate calibration block identifiers")
    if len({block.content_sha256 for block in blocks}) != expected_blocks:
        raise ValueError("duplicate calibration block contents")
    diagnostics: list[dict[str, Any]] = []
    selected: int | None = None
    for k in candidate_k:
        tail_weights: list[float] = []
        saturation: list[float] = []
        usable = True
        for block in blocks:
            thinned = thinned_snapshots(block, max(DROPOUT_LEVELS), seed)
            if min(snapshot.x.shape[0] for snapshot in thinned) < k:
                usable = False
                break
            model = StreamingCircularDPMM(
                max_clusters=k, alpha=alpha, heading_kappa=heading_kappa
            )
            for snapshot in block.snapshots:
                model.partial_fit(snapshot.x, snapshot.theta)
            weights = model.expected_weights()
            tail_weights.append(float(weights[-1].item()))
            saturation.append(float(model.active_clusters() >= k))
        q95 = float(np.quantile(tail_weights, 0.95)) if tail_weights else float("inf")
        saturation_rate = float(np.mean(saturation)) if saturation else 1.0
        passed = usable and q95 <= tail_threshold and saturation_rate <= saturation_threshold
        diagnostics.append(
            {
                "K": k,
                "usable": usable,
                "tail_weight_q95": q95,
                "saturation_rate": saturation_rate,
                "passed": passed,
            }
        )
        if passed and selected is None:
            selected = k
    if selected is None:
        raise RuntimeError(
            "no candidate K passed the residual/saturation rule; expand the grid "
            "and repeat calibration before formal execution"
        )

    locked = {
        "schema": LOCK_SCHEMA,
        "status": "locked",
        "calibration_block_ids": [block.block_id for block in blocks],
        "calibration_block_sha256": [block.content_sha256 for block in blocks],
        "selected_K": selected,
        "candidate_diagnostics": diagnostics,
        "tail_threshold": tail_threshold,
        "saturation_threshold": saturation_threshold,
        "alpha": alpha,
        "heading_kappa": heading_kappa,
        "particles": particles,
        "rejuvenation_window": 24,
        "dropout_levels": list(DROPOUT_LEVELS),
        "base_seed": seed,
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "secondary_endpoint": "category_nmi_evaluation_only",
    }
    locked["lock_sha256"] = _canonical_sha256(locked)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(locked, indent=2) + "\n", encoding="utf-8")
    return locked


def _fit_vi(
    snapshots: tuple[SnapshotData, ...], config: dict[str, Any]
) -> dict[str, Any]:
    model = StreamingCircularDPMM(
        max_clusters=int(config["selected_K"]),
        alpha=float(config["alpha"]),
        heading_kappa=float(config["heading_kappa"]),
    )
    labels: list[np.ndarray] = []
    categories: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    if not snapshots or snapshots[0].x.shape[0] < model.K:
        raise ValueError("initial retained snapshot has fewer observations than locked K")
    for index, snapshot in enumerate(snapshots):
        if snapshot.x.shape[0] < 1:
            raise ValueError("retained snapshot is empty")
        if index > 0:
            scores.append(model.log_predictive(snapshot.x, snapshot.theta).detach().cpu().numpy())
        responsibility = model.partial_fit(snapshot.x, snapshot.theta)
        labels.append(torch.argmax(responsibility, dim=1).detach().cpu().numpy())
        categories.append(snapshot.category)
    weights = model.expected_weights().detach().cpu().numpy()
    return {
        "labels": np.concatenate(labels),
        "categories": np.concatenate(categories),
        "scores": np.concatenate(scores) if scores else np.asarray([], dtype=float),
        "occupied_clusters": model.active_clusters(),
        "tail_weight": float(weights[-1]),
        "particle_ess": float("nan"),
    }


def _fit_smc(
    snapshots: tuple[SnapshotData, ...], config: dict[str, Any], seed: int
) -> dict[str, Any]:
    first = snapshots[0]
    if first.x.shape[0] < 2:
        raise ValueError("SMC calibration snapshot is empty")
    model = CollapsedDPSMC(
        first.x,
        particles=int(config["particles"]),
        alpha=float(config["alpha"]),
        heading_kappa=float(config["heading_kappa"]),
        rejuvenation_window=int(config["rejuvenation_window"]),
        seed=seed,
    )
    categories: list[np.ndarray] = []
    first_count = first.x.shape[0]
    for snapshot in snapshots:
        model.partial_fit(snapshot.x, snapshot.theta)
        categories.append(snapshot.category)
    labels = model.map_partition()
    scores = np.asarray(model.log_score_history[first_count:], dtype=float)
    return {
        "labels": labels,
        "categories": np.concatenate(categories),
        "scores": scores,
        "occupied_clusters": int(np.unique(labels).size),
        "tail_weight": float("nan"),
        "particle_ess": float(np.mean(model.ess_history)),
    }


def execute_cell(
    block: BlockData,
    method: str,
    dropout_rate: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    if method not in METHODS or dropout_rate not in DROPOUT_LEVELS:
        raise ValueError("cell is outside the locked 2 x 3 design")
    if block.phase != "formal":
        raise ValueError("formal execution cannot consume calibration blocks")
    snapshots = thinned_snapshots(block, dropout_rate, int(config["base_seed"]))
    retained = sum(snapshot.x.shape[0] for snapshot in snapshots)
    seed = _stable_seed(config["base_seed"], block.block_id, method, dropout_rate)
    fit = _fit_vi if method == "streaming_vi" else (
        lambda data, cfg: _fit_smc(data, cfg, seed)
    )
    result, elapsed_ms = synchronized_latency_ms(fit, snapshots, config)
    labels = np.asarray(result.pop("labels"))
    categories = np.asarray(result.pop("categories"))
    scores = np.asarray(result.pop("scores"), dtype=float)
    return {
        "block_id": block.block_id,
        "block_sha256": block.content_sha256,
        "method": method,
        "dropout_rate": dropout_rate,
        "retained_observations": retained,
        "scored_observations": int(scores.size),
        "mean_prequential_log_score": float(np.mean(scores)) if scores.size else float("nan"),
        "compute_latency_ms": elapsed_ms,
        "compute_latency_ms_per_observation": elapsed_ms / retained,
        "category_nmi_evaluation_only": normalized_mutual_information(labels, categories),
        **result,
        "device": str(DEVICE),
        "success": True,
    }


def _warm_up() -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(17)
    x = torch.randn(40, 2, dtype=DTYPE, device=DEVICE, generator=generator)
    theta = torch.remainder(torch.randn(40, dtype=DTYPE, device=DEVICE, generator=generator), 2 * math.pi)
    model = StreamingCircularDPMM(max_clusters=8)
    model.partial_fit(x, theta)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def run_factorial(
    block_paths: Iterable[Path],
    lock_path: Path,
    output: Path,
    expected_blocks: int = 100,
) -> list[dict[str, Any]]:
    config = load_locked_config(lock_path)
    blocks = [load_block(path) for path in sorted(block_paths)]
    if len(blocks) != expected_blocks:
        raise ValueError(f"formal run requires exactly {expected_blocks} verified blocks")
    if len({block.block_id for block in blocks}) != expected_blocks:
        raise ValueError("duplicate formal block identifiers")
    if len({block.content_sha256 for block in blocks}) != expected_blocks:
        raise ValueError("duplicate formal block contents")
    if any(block.phase != "formal" for block in blocks):
        raise ValueError("formal execution requires formal-phase blocks")
    calibration_ids = set(config["calibration_block_ids"])
    if calibration_ids.intersection(block.block_id for block in blocks):
        raise ValueError("calibration/formal block leakage detected")

    block_lookup = {block.block_id: block for block in blocks}
    expected_keys = {
        (block.block_id, method, dropout)
        for block in blocks
        for method in METHODS
        for dropout in DROPOUT_LEVELS
    }
    existing: dict[tuple[str, str, float], dict[str, Any]] = {}
    if output.exists():
        with output.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = _result_key(row)
                if key not in expected_keys:
                    raise ValueError(f"checkpoint contains out-of-design cell: {key}")
                if key in existing:
                    raise ValueError(f"checkpoint contains duplicate cell: {key}")
                _validate_result_row(
                    row,
                    expected_lock=str(config["lock_sha256"]),
                    expected_block_sha=block_lookup[key[0]].content_sha256,
                )
                existing[key] = row
    _warm_up()
    records: list[dict[str, Any]] = list(existing.values())
    for block in blocks:
        cells = [(method, dropout) for method in METHODS for dropout in DROPOUT_LEVELS]
        rng = np.random.default_rng(_stable_seed(config["base_seed"], block.block_id, "cell-order"))
        rng.shuffle(cells)
        for method, dropout in cells:
            key = (block.block_id, method, dropout)
            if key in existing:
                if existing[key].get("block_sha256") != block.content_sha256:
                    raise ValueError(
                        f"stale checkpoint for {block.block_id}: block hash changed"
                    )
                continue
            record = execute_cell(block, method, dropout, config)
            record["lock_sha256"] = config["lock_sha256"]
            records.append(record)
            _write_results(output, records)
    _write_results(output, records)
    return records


def _result_key(row: dict[str, Any]) -> tuple[str, str, float]:
    try:
        return str(row["block_id"]), str(row["method"]), float(row["dropout_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("result row has an invalid factorial key") from error


def _is_success(value: Any) -> bool:
    return str(value).lower() in {"true", "1"}


def _validate_result_row(
    row: dict[str, Any], expected_lock: str, expected_block_sha: str | None = None
) -> None:
    if row.get("lock_sha256") != expected_lock:
        raise ValueError("result row was not produced under the verified analysis lock")
    if expected_block_sha is not None and row.get("block_sha256") != expected_block_sha:
        raise ValueError(f"stale checkpoint for {row.get('block_id')}: block hash changed")
    if not _is_success(row.get("success")):
        raise ValueError("result row is not a successful completed cell")
    for endpoint in PRIMARY_ENDPOINTS:
        try:
            value = float(row[endpoint])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"result row has invalid primary endpoint: {endpoint}") from error
        if not math.isfinite(value):
            raise ValueError(f"result row has non-finite primary endpoint: {endpoint}")
        if endpoint != "mean_prequential_log_score" and value <= 0:
            raise ValueError(f"result row has non-positive primary endpoint: {endpoint}")


def _write_results(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def _mean_ci(values: np.ndarray) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return {
            "n": int(n),
            "estimate": float("nan"),
            "lower_95": float("nan"),
            "upper_95": float("nan"),
            "p_value_two_sided": float("nan"),
        }
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    critical = float(student_t.ppf(0.975, n - 1))
    if se == 0:
        p_value = 1.0 if mean == 0 else 0.0
    else:
        p_value = float(2.0 * student_t.sf(abs(mean / se), n - 1))
    return {
        "n": int(n),
        "estimate": mean,
        "lower_95": mean - critical * se,
        "upper_95": mean + critical * se,
        "p_value_two_sided": p_value,
    }


def _add_holm_adjustment(contrasts: Iterable[dict[str, Any]]) -> None:
    valid = [item for item in contrasts if math.isfinite(item["p_value_two_sided"])]
    ordered = sorted(valid, key=lambda item: item["p_value_two_sided"])
    running = 0.0
    total = len(ordered)
    for rank, item in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * item["p_value_two_sided"])
        running = max(running, adjusted)
        item["p_value_holm"] = running
    for item in contrasts:
        item.setdefault("p_value_holm", float("nan"))


def analyze_results(
    results_path: Path,
    lock_path: Path,
    public_summary: Path,
    project_status_path: Path,
    expected_blocks: int = 100,
) -> dict[str, Any]:
    config = load_locked_config(lock_path)
    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_rows = expected_blocks * len(METHODS) * len(DROPOUT_LEVELS)
    keys = {_result_key(row) for row in rows}
    if len(rows) != expected_rows or len(keys) != expected_rows:
        raise ValueError(f"analysis is locked until all {expected_rows} unique cells exist")

    block_ids = sorted({row["block_id"] for row in rows})
    if len(block_ids) != expected_blocks:
        raise ValueError(f"analysis requires exactly {expected_blocks} distinct blocks")
    expected_keys = {
        (block_id, method, dropout)
        for block_id in block_ids
        for method in METHODS
        for dropout in DROPOUT_LEVELS
    }
    if keys != expected_keys:
        raise ValueError("analysis contains cells outside the locked 2 x 3 design")
    if set(config["calibration_block_ids"]).intersection(block_ids):
        raise ValueError("calibration/formal block leakage detected during analysis")

    block_hashes: dict[str, str] = {}
    for row in rows:
        _validate_result_row(row, expected_lock=str(config["lock_sha256"]))
        block_hash = row.get("block_sha256")
        if (
            not isinstance(block_hash, str)
            or len(block_hash) != 64
            or any(character not in "0123456789abcdef" for character in block_hash)
        ):
            raise ValueError("result row has an invalid block fingerprint")
        prior = block_hashes.setdefault(row["block_id"], block_hash)
        if prior != block_hash:
            raise ValueError(f"result rows mix block contents for {row['block_id']}")

    lookup = {(row["block_id"], row["method"], float(row["dropout_rate"])): row for row in rows}
    endpoint_specs = {
        "mean_prequential_log_score": False,
        "log_compute_latency_ms_per_observation": True,
        "occupied_clusters": False,
        "category_nmi_evaluation_only": False,
    }
    analyses: dict[str, Any] = {}
    for endpoint, log_transform in endpoint_specs.items():
        cube = np.empty((expected_blocks, 2, 3), dtype=float)
        for b, block_id in enumerate(block_ids):
            for m, method in enumerate(METHODS):
                for d, dropout in enumerate(DROPOUT_LEVELS):
                    source = "compute_latency_ms_per_observation" if log_transform else endpoint
                    value = float(lookup[(block_id, method, dropout)][source])
                    if endpoint == "category_nmi_evaluation_only":
                        if math.isinf(value) or (math.isfinite(value) and not 0 <= value <= 1):
                            raise ValueError("category NMI must be in [0, 1] or missing")
                    cube[b, m, d] = math.log(value) if log_transform else value
        method_difference = cube[:, 1, :] - cube[:, 0, :]
        marginal = cube.mean(axis=1)
        paired = {
            str(dropout): _mean_ci(method_difference[:, index])
            for index, dropout in enumerate(DROPOUT_LEVELS)
        }
        orthogonal = {
            "architecture_average": _mean_ci(method_difference.mean(axis=1)),
            "dropout_linear": _mean_ci(marginal @ np.asarray([-1.0, 0.0, 1.0])),
            "dropout_quadratic": _mean_ci(marginal @ np.asarray([1.0, -2.0, 1.0])),
            "architecture_x_linear": _mean_ci(method_difference @ np.asarray([-1.0, 0.0, 1.0])),
            "architecture_x_quadratic": _mean_ci(method_difference @ np.asarray([1.0, -2.0, 1.0])),
        }
        _add_holm_adjustment([*paired.values(), *orthogonal.values()])
        analyses[endpoint] = {
            "direction": "SMC minus VI" if not log_transform else "log(SMC latency) minus log(VI latency)",
            "multiplicity": "Holm adjustment across the eight prespecified contrasts for this endpoint",
            "paired_architecture_by_dropout": paired,
            "orthogonal_block_contrasts": orthogonal,
        }

    results_sha256 = hashlib.sha256(results_path.read_bytes()).hexdigest()
    block_manifest_sha256 = _canonical_sha256(sorted(block_hashes.items()))
    summary = {
        "project": "OpenSky Streaming DPMM",
        "status": "complete_empirical_benchmark",
        "blocks": expected_blocks,
        "factorial_cells": 6,
        "executions": expected_rows,
        "methods": list(METHODS),
        "dropout_levels": list(DROPOUT_LEVELS),
        "analysis": analyses,
        "provenance": {
            "analysis_lock_sha256": config["lock_sha256"],
            "factorial_results_sha256": results_sha256,
            "formal_block_manifest_sha256": block_manifest_sha256,
        },
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(DEVICE),
        },
        "claim_boundary": (
            "Empirical OpenSky state-vector clustering under controlled thinning; "
            "not radar-pulse, emitter-identification, ELINT, SIGINT, or operational validation."
        ),
    }
    public_summary.parent.mkdir(parents=True, exist_ok=True)
    public_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    status = json.loads(project_status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "status": "empirical_benchmark_complete",
            "completed_blocks": expected_blocks,
            "completed_executions": expected_rows,
            "empirical_results_available": True,
        }
    )
    project_status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the frozen OpenSky experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--blocks", type=Path, default=Path("data/recorded_blocks"))
    calibration.add_argument("--output", type=Path, default=Path("data/results/locked_config.json"))

    run = subparsers.add_parser("run")
    run.add_argument("--blocks", type=Path, default=Path("data/recorded_blocks"))
    run.add_argument("--lock", type=Path, default=Path("data/results/locked_config.json"))
    run.add_argument("--output", type=Path, default=Path("data/results/factorial_results.csv"))

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--results", type=Path, default=Path("data/results/factorial_results.csv"))
    analyze.add_argument("--lock", type=Path, default=Path("data/results/locked_config.json"))
    analyze.add_argument("--public-summary", type=Path, default=Path("data/empirical_summary.json"))
    analyze.add_argument("--status", type=Path, default=Path("data/project_status.json"))
    args = parser.parse_args()

    if args.command == "calibrate":
        result = calibrate_truncation(args.blocks.glob("calibration-*.json.gz"), args.output)
    elif args.command == "run":
        result = {"executions": len(run_factorial(args.blocks.glob("formal-*.json.gz"), args.lock, args.output))}
    else:
        result = analyze_results(args.results, args.lock, args.public_summary, args.status)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
