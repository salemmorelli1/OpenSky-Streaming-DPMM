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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import scipy
from scipy.stats import t as student_t

from .collector import read_verified_block
from .contracts import (
    CALIBRATION_BLOCKS,
    FORMAL_BLOCKS,
    ENDPOINT_PROTOCOL,
    collection_contract,
    finite_real,
    integer,
)
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
LOCK_SCHEMA = "opensky-frozen-analysis-v2"
PRIMARY_ENDPOINTS = (
    "mean_prequential_log_score",
    "compute_latency_ms_per_observation",
    "occupied_clusters",
)
STRING_FIELDS = (
    "block_id",
    "block_sha256",
    "method",
    "device",
    "endpoint_protocol",
    "implementation_sha256",
    "execution_environment_sha256",
    "lock_sha256",
)
INTEGER_FIELDS = (
    "retained_observations",
    "scored_observations",
    "category_observations",
    "occupied_clusters",
)
FLOAT_FIELDS = (
    "dropout_rate",
    "mean_prequential_log_score",
    "compute_latency_ms",
    "compute_latency_ms_per_observation",
    "category_nmi_evaluation_only",
    "tail_weight",
    "particle_ess",
)
RESULT_FIELDS = (
    *STRING_FIELDS,
    *INTEGER_FIELDS,
    *FLOAT_FIELDS,
    "success",
    "result_sha256",
)
AMENDED_ACQUISITION = {
    "bbox": [49.0, 7.0, 54.0, 13.0],
    "snapshots_per_block": 6,
    "interval_seconds": 5.0,
    "minimum_eligible_states": 32,
    "max_attempts_per_block": 4,
    "retry_delay_seconds": 900.0,
}
CONFIRMATORY_K_GRID = (8, 12, 16, 24, 32)


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
    acquisition_contract: dict[str, Any] | None = None
    api_times: tuple[int, ...] = ()


def implementation_sha256() -> str:
    """Bind locks to the installed inference, acquisition, and replay code."""

    root = Path(__file__).resolve().parent
    return _canonical_sha256(
        {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "__init__.py",
                "contracts.py",
                "collector.py",
                "engine.py",
                "empirical.py",
            )
        }
    )


def execution_environment() -> dict[str, Any]:
    cpu_name = platform.processor()
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_name,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": str(DEVICE),
        "device_name": torch.cuda.get_device_name(DEVICE)
        if DEVICE.type == "cuda"
        else cpu_name,
        "cuda": torch.version.cuda,
        "dtype": str(DTYPE),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _stable_seed(*parts: object) -> int:
    value = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_locked_config(path: Path, *, require_runtime: bool = False) -> dict[str, Any]:
    """Load a lock only when its embedded fingerprint matches its contents."""

    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or config.get("schema") != LOCK_SCHEMA
        or config.get("status") != "locked"
    ):
        raise ValueError("invalid or unlocked analysis configuration")
    expected = config.get("lock_sha256")
    fingerprint_input = dict(config)
    fingerprint_input.pop("lock_sha256", None)
    actual = _canonical_sha256(fingerprint_input)
    if not isinstance(expected, str) or not hmac.compare_digest(expected, actual):
        raise ValueError("analysis lock fingerprint mismatch")
    if tuple(config.get("dropout_levels", ())) != DROPOUT_LEVELS:
        raise ValueError(
            "analysis lock does not match the implemented factorial design"
        )
    if tuple(config.get("primary_endpoints", ())) != PRIMARY_ENDPOINTS:
        raise ValueError("analysis lock does not match the primary endpoints")
    if config.get("endpoint_protocol") != ENDPOINT_PROTOCOL:
        raise ValueError("analysis lock uses obsolete endpoint semantics; recalibrate")
    if config.get("implementation_sha256") != implementation_sha256():
        raise ValueError("analysis lock uses another implementation; recalibrate")
    if config.get("scope") not in {"confirmatory", "validation_only"}:
        raise ValueError("analysis lock has invalid execution scope")
    integer(config.get("selected_K"), "selected_K", 2)
    integer(config.get("particles"), "particles", 2)
    integer(config.get("rejuvenation_window"), "rejuvenation_window")
    integer(config.get("base_seed"), "base_seed", 0, 2**32 - 1)
    finite_real(config.get("alpha"), "alpha", positive=True)
    finite_real(config.get("heading_kappa"), "heading_kappa", positive=True)
    means, variances = config.get("base_mean"), config.get("base_variance")
    if (
        not isinstance(means, list)
        or not isinstance(variances, list)
        or len(means) != 2
        or len(variances) != 2
    ):
        raise ValueError("analysis lock has no fixed Normal-Gamma base")
    for mean, variance in zip(means, variances):
        finite_real(mean, "base_mean", minimum=-math.inf)
        finite_real(variance, "base_variance", positive=True)
    ids = config.get("calibration_block_ids")
    hashes = config.get("calibration_block_sha256")
    count = CALIBRATION_BLOCKS if config["scope"] == "confirmatory" else len(ids or [])
    if (
        not isinstance(ids, list)
        or len(ids) != count
        or not ids
        or len(set(ids)) != count
    ):
        raise ValueError("invalid calibration block manifest")
    if any(
        not isinstance(item, str) or not re.fullmatch(r"calibration-[0-9]{4}", item)
        for item in ids
    ):
        raise ValueError("invalid calibration block identifiers")
    if (
        not isinstance(hashes, list)
        or len(hashes) != count
        or len(set(hashes)) != count
    ):
        raise ValueError("invalid calibration block fingerprint manifest")
    if any(
        not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item)
        for item in hashes
    ):
        raise ValueError("invalid calibration fingerprint")
    integer(config.get("calibration_api_time_end"), "calibration_api_time_end", 1)
    _validate_collection_contract(config.get("collection_contract"))
    if (
        config["scope"] == "confirmatory"
        and config["collection_contract"] != AMENDED_ACQUISITION
    ):
        raise ValueError(
            "confirmatory lock does not use the amended acquisition protocol"
        )
    runtime = config.get("execution_environment")
    if config["scope"] == "confirmatory":
        fixed = {
            "alpha": 1.5,
            "heading_kappa": 6.0,
            "particles": 64,
            "rejuvenation_window": 24,
            "base_seed": 2026,
            "tail_threshold": 0.01,
            "saturation_threshold": 0.05,
        }
        if any(config.get(key) != value for key, value in fixed.items()):
            raise ValueError("confirmatory lock changes the frozen model settings")
        diagnostics = config.get("candidate_diagnostics")
        if not isinstance(diagnostics, list) or not all(
            isinstance(item, dict) for item in diagnostics
        ):
            raise ValueError("confirmatory lock has no calibration diagnostics")
        if tuple(item.get("K") for item in diagnostics) != CONFIRMATORY_K_GRID:
            raise ValueError("confirmatory lock changes the frozen K grid")
        passed = [item["K"] for item in diagnostics if item.get("passed") is True]
        if not passed or config["selected_K"] != passed[0]:
            raise ValueError("locked K is not the first passing calibration candidate")
    if not isinstance(runtime, dict) or config.get(
        "execution_environment_sha256"
    ) != _canonical_sha256(runtime):
        raise ValueError("invalid execution environment fingerprint")
    if require_runtime and runtime != execution_environment():
        raise ValueError(
            "compute environment changed after calibration; recalibrate or restore it"
        )
    return config


def _validate_collection_contract(contract: Any) -> None:
    from .collector import CollectionConfig

    if not isinstance(contract, dict):
        raise ValueError("analysis lock has no acquisition protocol")
    try:
        candidate = CollectionConfig(phase="calibration", **contract)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid acquisition protocol") from error
    if candidate.contract() != contract:
        raise ValueError("acquisition protocol is not canonical")


def _validate_block_set(
    blocks: list[BlockData],
    phase: str,
    count: int,
    contract: dict[str, Any] | None = None,
    after: int = 0,
) -> None:
    if len(blocks) != count:
        raise ValueError(f"{phase} requires exactly {count} verified blocks")
    if any(block.phase != phase for block in blocks):
        raise ValueError(f"{phase} requires {phase}-phase blocks")
    if len({block.block_id for block in blocks}) != count:
        raise ValueError(f"duplicate {phase} block identifiers")
    if len({block.content_sha256 for block in blocks}) != count:
        raise ValueError(f"duplicate {phase} block contents")
    previous = after
    for block in blocks:
        if not re.fullmatch(rf"{phase}-[0-9]{{4}}", block.block_id):
            raise ValueError("block identifier does not match its phase")
        if contract is not None and block.acquisition_contract != contract:
            raise ValueError(
                "block uses a different acquisition protocol than the lock"
            )
        if len(block.api_times) != len(block.snapshots) or len(block.api_times) < 2:
            raise ValueError("block has no verified streaming timestamps")
        if block.api_times[0] <= previous or any(
            b <= a for a, b in zip(block.api_times, block.api_times[1:])
        ):
            raise ValueError(
                "block windows overlap, precede calibration, or are out of order"
            )
        previous = block.api_times[-1]


def load_block(path: Path) -> BlockData:
    payload = read_verified_block(path)
    snapshots: list[SnapshotData] = []
    for snapshot in payload["snapshots"]:
        observations = snapshot["observations"]
        x = torch.as_tensor(
            [
                [
                    math.log1p(float(row["velocity"])),
                    math.asinh(float(row["vertical_rate"]) / 5.0),
                ]
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
            [
                -1 if row.get("category") is None else int(row["category"])
                for row in observations
            ],
            dtype=np.int64,
        )
        snapshots.append(SnapshotData(x=x, theta=theta, category=category))
    return BlockData(
        block_id=str(payload["block_id"]),
        phase=str(payload["phase"]),
        content_sha256=str(payload["integrity"]["content_sha256"]),
        snapshots=tuple(snapshots),
        acquisition_contract=collection_contract(payload),
        api_times=tuple(snapshot["api_time"] for snapshot in payload["snapshots"]),
    )


def normalized_mutual_information(labels: np.ndarray, categories: np.ndarray) -> float:
    """Arithmetic-mean normalized mutual information, invariant to label names."""

    labels = np.asarray(labels)
    categories = np.asarray(categories)
    # OpenSky category 0 means "no information" and 1 means "no ADS-B
    # category information"; neither is an external class label.
    valid = (categories >= 2) & (categories <= 20) & (categories != 13)
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
    h_row = float(
        -np.sum(
            row_probability[row_probability > 0]
            * np.log(row_probability[row_probability > 0])
        )
    )
    h_column = float(
        -np.sum(
            column_probability[column_probability > 0]
            * np.log(column_probability[column_probability > 0])
        )
    )
    entropy_mean = 0.5 * (h_row + h_column)
    return (
        1.0
        if entropy_mean == 0 and np.array_equal(row, column)
        else (mutual_information / entropy_mean if entropy_mean > 0 else 0.0)
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
    *,
    validation_only: bool = False,
) -> dict[str, Any]:
    """Select K using calibration blocks that are excluded from formal analysis."""

    integer(expected_blocks, "expected_blocks", 1)
    if expected_blocks != CALIBRATION_BLOCKS and not validation_only:
        raise ValueError("confirmatory calibration requires exactly 10 blocks")
    if not candidate_k or tuple(sorted(set(candidate_k))) != candidate_k:
        raise ValueError("candidate K grid must be increasing and unique")
    for k in candidate_k:
        integer(k, "candidate K", 2)
    finite_real(alpha, "alpha", positive=True)
    finite_real(heading_kappa, "heading_kappa", positive=True)
    finite_real(tail_threshold, "tail_threshold")
    finite_real(saturation_threshold, "saturation_threshold")
    if tail_threshold >= 1 or saturation_threshold >= 1:
        raise ValueError("calibration thresholds must lie in [0, 1)")
    integer(particles, "particles", 2)
    integer(seed, "seed", 0, 2**32 - 1)
    if not validation_only and (
        candidate_k != CONFIRMATORY_K_GRID
        or (alpha, heading_kappa, particles, seed, tail_threshold, saturation_threshold)
        != (1.5, 6.0, 64, 2026, 0.01, 0.05)
    ):
        raise ValueError(
            "confirmatory calibration cannot change the frozen model settings"
        )
    blocks = [load_block(path) for path in sorted(block_paths)]
    _validate_block_set(blocks, "calibration", expected_blocks)
    contract = blocks[0].acquisition_contract
    _validate_collection_contract(contract)
    if not validation_only and contract != AMENDED_ACQUISITION:
        raise ValueError(
            "confirmatory calibration requires the amended acquisition protocol"
        )
    _validate_block_set(blocks, "calibration", expected_blocks, contract)
    calibration_x = torch.cat(
        [snapshot.x for block in blocks for snapshot in block.snapshots], dim=0
    )
    base_mean = calibration_x.mean(0).detach().cpu().tolist()
    base_variance = (
        calibration_x.var(0, unbiased=True)
        .clamp_min(torch.finfo(DTYPE).eps)
        .detach()
        .cpu()
        .tolist()
    )
    diagnostics: list[dict[str, Any]] = []
    selected: int | None = None
    for k in candidate_k:
        tail_weights: list[float] = []
        saturation: list[float] = []
        usable = True
        for block in blocks:
            thinned = thinned_snapshots(block, max(DROPOUT_LEVELS), seed)
            if thinned[0].x.shape[0] < k or any(
                snapshot.x.shape[0] < 1 for snapshot in thinned
            ):
                usable = False
                break
            block_tails, block_saturation = [], []
            for dropout in DROPOUT_LEVELS:
                model = StreamingCircularDPMM(
                    max_clusters=k,
                    alpha=alpha,
                    heading_kappa=heading_kappa,
                    base_mean=base_mean,
                    base_variance=base_variance,
                )
                for snapshot in thinned_snapshots(block, dropout, seed):
                    model.partial_fit(snapshot.x, snapshot.theta)
                block_tails.append(float(model.expected_weights()[-1].item()))
                block_saturation.append(float(model.active_clusters() >= k))
            tail_weights.append(max(block_tails))
            saturation.append(max(block_saturation))
        q95 = (
            float(np.quantile(tail_weights, 0.95)) if usable and tail_weights else None
        )
        saturation_rate = float(np.mean(saturation)) if usable and saturation else None
        passed = (
            usable
            and q95 is not None
            and q95 <= tail_threshold
            and saturation_rate <= saturation_threshold
        )
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
        "scope": "validation_only" if validation_only else "confirmatory",
        "endpoint_protocol": ENDPOINT_PROTOCOL,
        "implementation_sha256": implementation_sha256(),
        "collection_contract": contract,
        "calibration_api_time_end": blocks[-1].api_times[-1],
        "calibration_block_ids": [block.block_id for block in blocks],
        "calibration_block_sha256": [block.content_sha256 for block in blocks],
        "selected_K": selected,
        "candidate_diagnostics": diagnostics,
        "tail_threshold": tail_threshold,
        "saturation_threshold": saturation_threshold,
        "alpha": alpha,
        "heading_kappa": heading_kappa,
        "base_mean": base_mean,
        "base_variance": base_variance,
        "particles": particles,
        "rejuvenation_window": 24,
        "dropout_levels": list(DROPOUT_LEVELS),
        "base_seed": seed,
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "secondary_endpoint": "category_nmi_evaluation_only",
    }
    locked["execution_environment"] = execution_environment()
    locked["execution_environment_sha256"] = _canonical_sha256(
        locked["execution_environment"]
    )
    locked["lock_sha256"] = _canonical_sha256(locked)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(
            "refusing to overwrite an existing lock; use a new isolated path"
        )
    output.write_text(
        json.dumps(locked, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return locked


def _fit_vi(
    snapshots: tuple[SnapshotData, ...], config: dict[str, Any]
) -> dict[str, Any]:
    model = StreamingCircularDPMM(
        max_clusters=int(config["selected_K"]),
        alpha=float(config["alpha"]),
        heading_kappa=float(config["heading_kappa"]),
        base_mean=config.get("base_mean"),
        base_variance=config.get("base_variance"),
    )
    labels: list[np.ndarray] = []
    categories: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    if not snapshots or snapshots[0].x.shape[0] < model.K:
        raise ValueError(
            "initial retained snapshot has fewer observations than locked K"
        )
    for index, snapshot in enumerate(snapshots):
        if snapshot.x.shape[0] < 1:
            raise ValueError("retained snapshot is empty")
        if index > 0:
            scores.append(
                model.log_predictive(snapshot.x, snapshot.theta).detach().cpu().numpy()
            )
        model.partial_fit(snapshot.x, snapshot.theta)
        categories.append(snapshot.category)
    for snapshot in snapshots:
        labels.append(
            torch.argmax(model.responsibilities(snapshot.x, snapshot.theta), dim=1)
            .detach()
            .cpu()
            .numpy()
        )
    weights = model.expected_weights().detach().cpu().numpy()
    return {
        "labels": np.concatenate(labels),
        "categories": np.concatenate(categories),
        "scores": np.concatenate(scores) if scores else np.asarray([], dtype=float),
        "occupied_clusters": int(np.unique(np.concatenate(labels)).size),
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
        base_mean=config.get("base_mean"),
        base_variance=config.get("base_variance"),
        rejuvenation_window=int(config["rejuvenation_window"]),
        seed=seed,
    )
    categories: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for index, snapshot in enumerate(snapshots):
        if index > 0:
            scores.append(
                model.log_predictive(snapshot.x, snapshot.theta).detach().cpu().numpy()
            )
        model.partial_fit(snapshot.x, snapshot.theta)
        categories.append(snapshot.category)
    labels = model.map_partition()
    return {
        "labels": labels,
        "categories": np.concatenate(categories),
        "scores": np.concatenate(scores) if scores else np.asarray([], dtype=float),
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
    fit = (
        _fit_vi
        if method == "streaming_vi"
        else (lambda data, cfg: _fit_smc(data, cfg, seed))
    )
    result, elapsed_ms = synchronized_latency_ms(fit, snapshots, config)
    labels = np.asarray(result.pop("labels"))
    categories = np.asarray(result.pop("categories"))
    scores = np.asarray(result.pop("scores"), dtype=float)
    if not scores.size or not np.isfinite(scores).all():
        raise ValueError(
            "all scored observations must have finite pre-snapshot densities"
        )
    return {
        "block_id": block.block_id,
        "block_sha256": block.content_sha256,
        "method": method,
        "dropout_rate": dropout_rate,
        "retained_observations": retained,
        "scored_observations": int(scores.size),
        "category_observations": int(
            np.count_nonzero(
                (categories >= 2) & (categories <= 20) & (categories != 13)
            )
        ),
        "mean_prequential_log_score": float(np.mean(scores))
        if scores.size
        else float("nan"),
        "compute_latency_ms": elapsed_ms,
        "compute_latency_ms_per_observation": elapsed_ms / retained,
        "category_nmi_evaluation_only": normalized_mutual_information(
            labels, categories
        ),
        **result,
        "device": str(DEVICE),
        "endpoint_protocol": ENDPOINT_PROTOCOL,
        "implementation_sha256": config.get(
            "implementation_sha256", implementation_sha256()
        ),
        "execution_environment_sha256": config.get(
            "execution_environment_sha256", _canonical_sha256(execution_environment())
        ),
        "success": True,
    }


def _warm_up() -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(17)
    x = torch.randn(40, 2, dtype=DTYPE, device=DEVICE, generator=generator)
    theta = torch.remainder(
        torch.randn(40, dtype=DTYPE, device=DEVICE, generator=generator), 2 * math.pi
    )
    model = StreamingCircularDPMM(max_clusters=8)
    model.partial_fit(x, theta)
    model.log_predictive(x[:4], theta[:4])
    smc = CollapsedDPSMC(x[:8], particles=4, rejuvenation_window=4, seed=17)
    smc.partial_fit(x[:8], theta[:8])
    smc.log_predictive(x[:4], theta[:4])
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def run_factorial(
    block_paths: Iterable[Path],
    lock_path: Path,
    output: Path,
    expected_blocks: int = 100,
    *,
    validation_only: bool = False,
) -> list[dict[str, Any]]:
    integer(expected_blocks, "expected_blocks", 1)
    if expected_blocks != FORMAL_BLOCKS and not validation_only:
        raise ValueError("confirmatory execution requires exactly 100 blocks")
    config = load_locked_config(lock_path, require_runtime=True)
    if config["scope"] != ("validation_only" if validation_only else "confirmatory"):
        raise ValueError("execution scope does not match the calibration lock")
    blocks = [load_block(path) for path in sorted(block_paths)]
    _validate_block_set(
        blocks,
        "formal",
        expected_blocks,
        config["collection_contract"],
        config["calibration_api_time_end"],
    )
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
        for row in _read_results(output):
            key = _result_key(row)
            if key not in expected_keys:
                raise ValueError(f"checkpoint contains out-of-design cell: {key}")
            if key in existing:
                raise ValueError(f"checkpoint contains duplicate cell: {key}")
            _validate_result_row(row, config=config, block=block_lookup[key[0]])
            existing[key] = row
    _warm_up()
    records: list[dict[str, Any]] = list(existing.values())
    for block in blocks:
        cells = [(method, dropout) for method in METHODS for dropout in DROPOUT_LEVELS]
        rng = np.random.default_rng(
            _stable_seed(config["base_seed"], block.block_id, "cell-order")
        )
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
            record["result_sha256"] = result_sha256(record)
            _validate_result_row(record, config=config, block=block)
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


def _numeric_integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, str):
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError(f"result row has invalid count: {name}")
        value = int(value)
    return integer(value, name, minimum)


def result_sha256(row: dict[str, Any]) -> str:
    """Canonicalize typed CSV fields so serialization preserves their digest."""

    normalized: dict[str, Any] = {key: str(row[key]) for key in STRING_FIELDS}
    normalized.update({key: _numeric_integer(row[key], key) for key in INTEGER_FIELDS})
    for key in FLOAT_FIELDS:
        value = float(row[key])
        normalized[key] = None if math.isnan(value) else value
    normalized["success"] = _is_success(row.get("success"))
    return _canonical_sha256(normalized)


def _read_results(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(RESULT_FIELDS) or set(fields) != set(RESULT_FIELDS):
            raise ValueError(
                "result CSV has obsolete, missing, duplicate, or unknown columns"
            )
        rows = list(reader)
    if any(set(row) != set(RESULT_FIELDS) or None in row.values() for row in rows):
        raise ValueError("result CSV contains a malformed row")
    return rows


def _validate_result_row(
    row: dict[str, Any], config: dict[str, Any], block: BlockData
) -> None:
    if set(row) != set(RESULT_FIELDS):
        raise ValueError("result row has obsolete, missing, or unknown fields")
    if row.get("lock_sha256") != config["lock_sha256"]:
        raise ValueError("result row was not produced under the verified analysis lock")
    if row.get("block_sha256") != block.content_sha256:
        raise ValueError(
            f"stale checkpoint for {row.get('block_id')}: block hash changed"
        )
    if not _is_success(row.get("success")):
        raise ValueError("result row is not a successful completed cell")
    for field in (
        "endpoint_protocol",
        "implementation_sha256",
        "execution_environment_sha256",
    ):
        if row.get(field) != config[field]:
            raise ValueError(f"result row has incompatible {field}")
    if row.get("device") != config["execution_environment"]["device"]:
        raise ValueError("result row was computed on another device")
    _, method, dropout = _result_key(row)
    if (
        method not in METHODS
        or dropout not in DROPOUT_LEVELS
        or row["block_id"] != block.block_id
    ):
        raise ValueError("result row is outside the verified factorial design")
    for endpoint in PRIMARY_ENDPOINTS:
        try:
            value = float(row[endpoint])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"result row has invalid primary endpoint: {endpoint}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"result row has non-finite primary endpoint: {endpoint}")
        if endpoint != "mean_prequential_log_score" and value <= 0:
            raise ValueError(
                f"result row has non-positive primary endpoint: {endpoint}"
            )
    counts = {key: _numeric_integer(row[key], key) for key in INTEGER_FIELDS}
    retained_counts = [
        int(
            nested_keep_mask(
                snapshot.x.shape[0],
                dropout,
                _stable_seed(config["base_seed"], block.block_id, index),
            ).sum()
        )
        for index, snapshot in enumerate(block.snapshots)
    ]
    retained = sum(retained_counts)
    scored = sum(retained_counts[1:])
    if (
        counts["retained_observations"] != retained
        or counts["scored_observations"] != scored
        or scored < 1
    ):
        raise ValueError("result row counts do not match the frozen thinning masks")
    if not 1 <= counts["occupied_clusters"] <= retained:
        raise ValueError("result row has impossible occupied-cluster count")
    if method == "streaming_vi" and counts["occupied_clusters"] > config["selected_K"]:
        raise ValueError("VI occupancy exceeds the locked truncation")
    elapsed = float(row["compute_latency_ms"])
    per_observation = float(row["compute_latency_ms_per_observation"])
    if (
        not math.isfinite(elapsed)
        or elapsed <= 0
        or not math.isclose(elapsed / retained, per_observation, rel_tol=1e-10)
    ):
        raise ValueError(
            "result row latency is inconsistent with its observation count"
        )
    snapshots = thinned_snapshots(block, dropout, config["base_seed"])
    category_count = sum(
        int(
            np.count_nonzero(
                (s.category >= 2) & (s.category <= 20) & (s.category != 13)
            )
        )
        for s in snapshots
    )
    if counts["category_observations"] != category_count:
        raise ValueError(
            "result row category coverage does not match the recorded block"
        )
    nmi = float(row["category_nmi_evaluation_only"])
    if category_count < 2:
        if not math.isnan(nmi):
            raise ValueError(
                "category NMI must be missing when fewer than two labels exist"
            )
    elif not math.isfinite(nmi) or not 0 <= nmi <= 1:
        raise ValueError("category NMI must be finite and in [0, 1]")
    tail, ess = float(row["tail_weight"]), float(row["particle_ess"])
    if method == "streaming_vi":
        if not math.isfinite(tail) or not 0 <= tail <= 1 or not math.isnan(ess):
            raise ValueError("invalid VI health metrics")
    elif (
        not math.isnan(tail)
        or not math.isfinite(ess)
        or not 1 - 1e-10 <= ess <= config["particles"] + 1e-10
    ):
        raise ValueError("invalid SMC health metrics")
    if row.get("result_sha256") != result_sha256(row):
        raise ValueError("result row fingerprint mismatch")


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
    *,
    block_paths: Iterable[Path] | None = None,
    validation_only: bool = False,
) -> dict[str, Any]:
    integer(expected_blocks, "expected_blocks", 1)
    if expected_blocks != FORMAL_BLOCKS and not validation_only:
        raise ValueError("public analysis requires exactly 100 blocks and 600 cells")
    config = load_locked_config(lock_path)
    if config["scope"] != ("validation_only" if validation_only else "confirmatory"):
        raise ValueError("analysis scope does not match the calibration lock")
    rows = _read_results(results_path)
    expected_rows = expected_blocks * len(METHODS) * len(DROPOUT_LEVELS)
    keys = {_result_key(row) for row in rows}
    if len(rows) != expected_rows or len(keys) != expected_rows:
        raise ValueError(
            f"analysis is locked until all {expected_rows} unique cells exist"
        )

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
    if block_paths is None:
        raise ValueError("analysis requires the original verified formal blocks")
    blocks = [load_block(path) for path in sorted(block_paths)]
    _validate_block_set(
        blocks,
        "formal",
        expected_blocks,
        config["collection_contract"],
        config["calibration_api_time_end"],
    )
    block_lookup = {block.block_id: block for block in blocks}
    if set(block_ids) != set(block_lookup):
        raise ValueError("analysis CSV and verified formal block set do not match")

    block_hashes: dict[str, str] = {}
    for row in rows:
        _validate_result_row(row, config=config, block=block_lookup[row["block_id"]])
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

    lookup = {
        (row["block_id"], row["method"], float(row["dropout_rate"])): row
        for row in rows
    }
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
                    source = (
                        "compute_latency_ms_per_observation"
                        if log_transform
                        else endpoint
                    )
                    value = float(lookup[(block_id, method, dropout)][source])
                    if endpoint == "category_nmi_evaluation_only":
                        if math.isinf(value) or (
                            math.isfinite(value) and not 0 <= value <= 1
                        ):
                            raise ValueError(
                                "category NMI must be in [0, 1] or missing"
                            )
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
            "architecture_x_linear": _mean_ci(
                method_difference @ np.asarray([-1.0, 0.0, 1.0])
            ),
            "architecture_x_quadratic": _mean_ci(
                method_difference @ np.asarray([1.0, -2.0, 1.0])
            ),
        }
        _add_holm_adjustment([*paired.values(), *orthogonal.values()])
        analyses[endpoint] = {
            "direction": "SMC minus VI"
            if not log_transform
            else "log(SMC latency) minus log(VI latency)",
            "multiplicity": "Holm adjustment across the eight prespecified contrasts for this endpoint",
            "paired_architecture_by_dropout": paired,
            "orthogonal_block_contrasts": orthogonal,
        }

    results_sha256 = hashlib.sha256(results_path.read_bytes()).hexdigest()
    block_manifest_sha256 = _canonical_sha256(sorted(block_hashes.items()))
    summary = {
        "project": "OpenSky Streaming DPMM",
        "status": "software_validation_only"
        if validation_only
        else "complete_empirical_benchmark",
        "empirical_results_available": not validation_only,
        "endpoint_protocol": ENDPOINT_PROTOCOL,
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
            "implementation_sha256": config["implementation_sha256"],
            "execution_environment_sha256": config["execution_environment_sha256"],
        },
        "hardware": config["execution_environment"],
        "category_coverage": {
            method: {
                str(dropout): {
                    "retained_observations": sum(
                        int(
                            lookup[(block_id, method, dropout)]["retained_observations"]
                        )
                        for block_id in block_ids
                    ),
                    "category_observations": sum(
                        int(
                            lookup[(block_id, method, dropout)]["category_observations"]
                        )
                        for block_id in block_ids
                    ),
                }
                for dropout in DROPOUT_LEVELS
            }
            for method in METHODS
        },
        "claim_boundary": (
            "Empirical OpenSky state-vector clustering under controlled thinning; "
            "not radar-pulse, emitter-identification, ELINT, SIGINT, or operational validation."
        ),
    }
    summary = _json_safe(summary)
    public_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    public_summary.write_text(summary_text, encoding="utf-8")

    if validation_only:
        return summary

    status = json.loads(project_status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "status": "empirical_benchmark_complete",
            "completed_blocks": expected_blocks,
            "completed_executions": expected_rows,
            "empirical_results_available": True,
            "endpoint_protocol": ENDPOINT_PROTOCOL,
            "analysis_lock_sha256": config["lock_sha256"],
            "implementation_sha256": config["implementation_sha256"],
            "factorial_results_sha256": results_sha256,
            "formal_block_manifest_sha256": block_manifest_sha256,
            "empirical_summary_sha256": hashlib.sha256(
                public_summary.read_bytes()
            ).hexdigest(),
        }
    )
    project_status_path.write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the frozen OpenSky experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument(
        "--blocks", type=Path, default=Path("data/recorded_blocks")
    )
    calibration.add_argument(
        "--output", type=Path, default=Path("data/results/locked_config.json")
    )

    run = subparsers.add_parser("run")
    run.add_argument("--blocks", type=Path, default=Path("data/recorded_blocks"))
    run.add_argument(
        "--lock", type=Path, default=Path("data/results/locked_config.json")
    )
    run.add_argument(
        "--output", type=Path, default=Path("data/results/factorial_results.csv")
    )

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--blocks", type=Path, default=Path("data/recorded_blocks"))
    analyze.add_argument(
        "--results", type=Path, default=Path("data/results/factorial_results.csv")
    )
    analyze.add_argument(
        "--lock", type=Path, default=Path("data/results/locked_config.json")
    )
    analyze.add_argument(
        "--public-summary", type=Path, default=Path("data/empirical_summary.json")
    )
    analyze.add_argument(
        "--status", type=Path, default=Path("data/project_status.json")
    )
    args = parser.parse_args()

    if args.command == "calibrate":
        result = calibrate_truncation(
            args.blocks.glob("calibration-*.json.gz"), args.output
        )
    elif args.command == "run":
        result = {
            "executions": len(
                run_factorial(
                    args.blocks.glob("formal-*.json.gz"), args.lock, args.output
                )
            )
        }
    else:
        result = analyze_results(
            args.results,
            args.lock,
            args.public_summary,
            args.status,
            block_paths=args.blocks.glob("formal-*.json.gz"),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
