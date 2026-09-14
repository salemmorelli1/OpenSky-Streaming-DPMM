"""Restartable, privacy-minimized OpenSky acquisition for empirical blocks.

The collector intentionally does not import PyTorch.  Acquisition therefore
remains available when an enterprise Windows policy temporarily prevents a
numerical backend DLL from loading.  OAuth credentials are read from process
environment variables and are never serialized.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

API_URL = "https://opensky-network.org/api/states/all"
TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)
SCHEMA_VERSION = "opensky-empirical-block-v1"
DEFAULT_BBOX = (49.0, 7.0, 54.0, 13.0)
ICAO24_PATTERN = re.compile(r"^[0-9a-fA-F]{6}$")
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_IDENTITY_KEYS = {"icao24", "callsign", "origin_country", "squawk"}
REQUIRED_MODEL_FIELDS = ("velocity", "true_track", "vertical_rate")


class CollectionError(RuntimeError):
    """Raised when a block cannot be collected or verified."""


class EligibilityError(CollectionError):
    """Raised when a snapshot fails the prespecified completeness threshold."""

    def __init__(
        self,
        block_id: str,
        snapshot_index: int,
        eligible_states: int,
        minimum_eligible_states: int,
        eligible_counts: list[int],
    ) -> None:
        super().__init__(
            f"{block_id} snapshot {snapshot_index} has only "
            f"{eligible_states} eligible states"
        )
        self.block_id = block_id
        self.snapshot_index = snapshot_index
        self.eligible_states = eligible_states
        self.minimum_eligible_states = minimum_eligible_states
        self.eligible_counts = eligible_counts


@dataclass(frozen=True)
class CollectionConfig:
    phase: str
    bbox: tuple[float, float, float, float]
    snapshots_per_block: int
    interval_seconds: float
    minimum_eligible_states: int
    max_attempts_per_block: int = 1
    retry_delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.phase not in {"calibration", "formal"}:
            raise ValueError("phase must be calibration or formal")
        lamin, lomin, lamax, lomax = self.bbox
        if not (-90 <= lamin < lamax <= 90 and -180 <= lomin < lomax <= 180):
            raise ValueError("bbox must be (lamin, lomin, lamax, lomax)")
        if self.snapshots_per_block < 2:
            raise ValueError("a streaming block requires at least two snapshots")
        if self.interval_seconds < 0 or self.minimum_eligible_states < 1:
            raise ValueError("invalid timing or eligibility threshold")
        if self.max_attempts_per_block < 1 or self.retry_delay_seconds < 0:
            raise ValueError("invalid bounded-retry policy")


class OAuthSession:
    """Minimal OAuth2 client with rate-limit-aware requests."""

    def __init__(
        self,
        client_id: str | None,
        client_secret: str | None,
        timeout_seconds: float = 20.0,
        session: requests.Session | None = None,
    ) -> None:
        if (client_id is None) != (client_secret is None):
            raise ValueError("provide both OpenSky OAuth values or neither")
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self._token: str | None = None
        self._expires_at = 0.0

    def _headers(self) -> dict[str, str]:
        if self.client_id is None:
            return {}
        if self._token is None or time.time() >= self._expires_at - 30.0:
            response = self.session.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            self._token = str(payload["access_token"])
            self._expires_at = time.time() + float(payload.get("expires_in", 1800))
        return {"Authorization": f"Bearer {self._token}"}

    def get_states(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        lamin, lomin, lamax, lomax = bbox
        started = time.perf_counter()
        response = self.session.get(
            API_URL,
            params={
                "lamin": lamin,
                "lomin": lomin,
                "lamax": lamax,
                "lomax": lomax,
                "extended": 1,
            },
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        if response.status_code == 429:
            retry = response.headers.get(
                "X-Rate-Limit-Retry-After-Seconds",
                response.headers.get("Retry-After", "unknown"),
            )
            raise CollectionError(f"OpenSky rate limit reached; retry after {retry} s")
        response.raise_for_status()
        metadata = {
            "network_latency_ms": elapsed_ms,
            "rate_limit_remaining": _safe_int(
                response.headers.get("X-Rate-Limit-Remaining")
            ),
            "rate_limit_retry_after_seconds": _safe_int(
                response.headers.get("X-Rate-Limit-Retry-After-Seconds")
            ),
        }
        return response.json(), metadata


def _safe_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_or_create_salt(path: Path) -> bytes:
    """Create a stable local pseudonym key with owner-only permissions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        salt = path.read_bytes()
        if len(salt) < 32:
            raise CollectionError("pseudonym salt is shorter than 32 bytes")
        return salt
    salt = secrets.token_bytes(32)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        os.write(descriptor, salt)
        os.close(descriptor)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise
    return salt


def pseudonymize(icao24: str, salt: bytes) -> str:
    return hmac.new(salt, icao24.encode("ascii"), hashlib.sha256).hexdigest()[:24]


def sanitize_payload(payload: dict[str, Any], salt: bytes) -> dict[str, Any]:
    """Remove direct identity fields before any block is written to disk."""

    observations: list[dict[str, Any]] = []
    rejected = 0
    for row in payload.get("states") or []:
        if len(row) < 12:
            rejected += 1
            continue
        icao24, velocity, track, vertical_rate = row[0], row[9], row[10], row[11]
        if None in (icao24, velocity, track, vertical_rate):
            rejected += 1
            continue
        icao24 = str(icao24)
        if not ICAO24_PATTERN.fullmatch(icao24):
            rejected += 1
            continue
        try:
            velocity = float(velocity)
            track = float(track)
            vertical_rate = float(vertical_rate)
        except (TypeError, ValueError, OverflowError):
            rejected += 1
            continue
        if not all(math.isfinite(value) for value in (velocity, track, vertical_rate)):
            rejected += 1
            continue
        if velocity < 0:
            rejected += 1
            continue
        category = row[17] if len(row) > 17 else None
        if category is not None:
            try:
                category = int(category)
            except (TypeError, ValueError, OverflowError):
                category = None
        observations.append(
            {
                "track_key": pseudonymize(icao24.lower(), salt),
                "time_position": row[3],
                "last_contact": row[4],
                "longitude": row[5],
                "latitude": row[6],
                "baro_altitude": row[7],
                "on_ground": row[8],
                "velocity": velocity,
                "true_track": track % 360.0,
                "vertical_rate": vertical_rate,
                "geo_altitude": row[13] if len(row) > 13 else None,
                "position_source": row[16] if len(row) > 16 else None,
                "category": category,
            }
        )
    return {
        "api_time": int(payload.get("time", time.time())),
        "eligible_observations": len(observations),
        "rejected_incomplete_states": rejected,
        "observations": observations,
    }


def atomic_write_gzip_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = canonical_json_bytes(value)
    digest = sha256_bytes(canonical)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    os.close(descriptor)
    try:
        with Path(temporary).open("wb") as raw_stream, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_stream, mtime=0
        ) as stream:
            stream.write(canonical)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return digest


def read_verified_block(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        data = stream.read()
    value = json.loads(data)
    expected = value.get("integrity", {}).get("content_sha256")
    if not isinstance(expected, str) or not HEX64_PATTERN.fullmatch(expected):
        raise CollectionError(f"missing content checksum: {path}")
    copy = dict(value)
    copy["integrity"] = dict(value["integrity"])
    copy["integrity"].pop("content_sha256", None)
    actual = sha256_bytes(canonical_json_bytes(copy))
    if not hmac.compare_digest(expected, actual):
        raise CollectionError(f"checksum mismatch: {path}")
    _validate_block_structure(value, path)
    return value


def _validate_block_structure(value: Any, path: Path) -> None:
    """Fail closed on self-consistent but malformed empirical artifacts."""

    label = str(path)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise CollectionError(f"invalid block schema: {label}")
    phase = value.get("phase")
    block_id = value.get("block_id")
    if phase not in {"calibration", "formal"} or not isinstance(block_id, str):
        raise CollectionError(f"invalid block identity: {label}")
    if not re.fullmatch(rf"{phase}-[0-9]{{4}}", block_id):
        raise CollectionError(f"block id does not match phase: {label}")

    bbox = value.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise CollectionError(f"invalid block bbox: {label}")
    try:
        lamin, lomin, lamax, lomax = map(float, bbox)
    except (TypeError, ValueError, OverflowError) as error:
        raise CollectionError(f"invalid block bbox: {label}") from error
    if not (-90 <= lamin < lamax <= 90 and -180 <= lomin < lomax <= 180):
        raise CollectionError(f"invalid block bbox: {label}")

    snapshots = value.get("snapshots")
    count = value.get("snapshots_per_block")
    if not isinstance(count, int) or count < 2 or not isinstance(snapshots, list):
        raise CollectionError(f"invalid snapshot count: {label}")
    if len(snapshots) != count:
        raise CollectionError(f"snapshot count mismatch: {label}")
    policy = value.get("acceptance_policy")
    if not isinstance(policy, dict):
        raise CollectionError(f"missing acceptance policy: {label}")
    minimum = policy.get("minimum_eligible_states")
    if not isinstance(minimum, int) or minimum < 1:
        raise CollectionError(f"invalid eligibility policy: {label}")

    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, dict) or snapshot.get("snapshot_index") != index:
            raise CollectionError(f"invalid snapshot index in {label}")
        observations = snapshot.get("observations")
        eligible = snapshot.get("eligible_observations")
        if not isinstance(observations, list) or eligible != len(observations):
            raise CollectionError(f"eligible count mismatch in {label}")
        if eligible < minimum:
            raise CollectionError(f"snapshot violates eligibility policy: {label}")
        for observation in observations:
            if not isinstance(observation, dict):
                raise CollectionError(f"invalid observation in {label}")
            if FORBIDDEN_IDENTITY_KEYS.intersection(observation):
                raise CollectionError(f"direct identity field found in {label}")
            track_key = observation.get("track_key")
            if not isinstance(track_key, str) or not re.fullmatch(r"[0-9a-f]{24}", track_key):
                raise CollectionError(f"invalid pseudonymous track key in {label}")
            try:
                model_values = [float(observation[field]) for field in REQUIRED_MODEL_FIELDS]
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise CollectionError(f"invalid model observation in {label}") from error
            if not all(math.isfinite(item) for item in model_values) or model_values[0] < 0:
                raise CollectionError(f"non-finite model observation in {label}")

    integrity = value.get("integrity")
    raw_hashes = integrity.get("raw_response_sha256") if isinstance(integrity, dict) else None
    if not isinstance(raw_hashes, list) or len(raw_hashes) != count:
        raise CollectionError(f"invalid raw-response hash manifest: {label}")
    if any(not isinstance(item, str) or not HEX64_PATTERN.fullmatch(item) for item in raw_hashes):
        raise CollectionError(f"invalid raw-response hash: {label}")


def append_attempt_record(path: Path, record: dict[str, Any]) -> None:
    """Append one non-identifying acquisition-attempt record to a JSONL audit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def build_block(
    client: OAuthSession,
    config: CollectionConfig,
    salt: bytes,
    block_id: str,
    attempt_number: int = 1,
) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []
    raw_hashes: list[str] = []
    for index in range(config.snapshots_per_block):
        raw, transport = client.get_states(config.bbox)
        raw_hashes.append(sha256_bytes(canonical_json_bytes(raw)))
        sanitized = sanitize_payload(raw, salt)
        if sanitized["eligible_observations"] < config.minimum_eligible_states:
            eligible = int(sanitized["eligible_observations"])
            raise EligibilityError(
                block_id,
                index,
                eligible,
                config.minimum_eligible_states,
                [
                    int(snapshot["eligible_observations"])
                    for snapshot in snapshots
                ]
                + [eligible],
            )
        sanitized["snapshot_index"] = index
        sanitized["transport"] = transport
        snapshots.append(sanitized)
        if index + 1 < config.snapshots_per_block and config.interval_seconds:
            time.sleep(config.interval_seconds)

    block: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_id": block_id,
        "phase": config.phase,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "OpenSky Network /api/states/all",
        "bbox": list(config.bbox),
        "snapshots_per_block": config.snapshots_per_block,
        "interval_seconds": config.interval_seconds,
        "acceptance_policy": {
            "minimum_eligible_states": config.minimum_eligible_states,
            "max_attempts_per_block": config.max_attempts_per_block,
            "retry_delay_seconds": config.retry_delay_seconds,
            "accepted_attempt": attempt_number,
        },
        "identity_policy": (
            "ICAO24 transformed by a local keyed HMAC; callsign, origin country, "
            "squawk, and raw ICAO24 are discarded before serialization."
        ),
        "snapshots": snapshots,
        "integrity": {
            "raw_response_sha256": raw_hashes,
            "content_sha256": None,
        },
    }
    integrity_input = dict(block)
    integrity_input["integrity"] = dict(block["integrity"])
    integrity_input["integrity"].pop("content_sha256")
    block["integrity"]["content_sha256"] = sha256_bytes(
        canonical_json_bytes(integrity_input)
    )
    return block


def collect_blocks(
    output_dir: Path,
    count: int,
    config: CollectionConfig,
    spacing_seconds: float,
    start_index: int = 1,
    client: OAuthSession | None = None,
    salt_path: Path | None = None,
    attempt_log_path: Path | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Collect missing block indices and verify every completed artifact."""

    if count < 1 or start_index < 1 or spacing_seconds < 0:
        raise ValueError("invalid block count, start index, or spacing")
    client = client or OAuthSession(
        os.getenv("OPENSKY_CLIENT_ID"), os.getenv("OPENSKY_CLIENT_SECRET")
    )
    salt = load_or_create_salt(
        salt_path or Path("data/private/pseudonym_salt.bin")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[Path] = []
    indices = range(start_index, start_index + count)
    for offset, block_index in enumerate(indices):
        block_id = f"{config.phase}-{block_index:04d}"
        destination = output_dir / f"{block_id}.json.gz"
        if destination.exists():
            read_verified_block(destination)
            completed.append(destination)
            continue

        block: dict[str, Any] | None = None
        for attempt in range(1, config.max_attempts_per_block + 1):
            attempted_at = datetime.now(timezone.utc).isoformat()
            try:
                block = build_block(client, config, salt, block_id, attempt)
            except (CollectionError, requests.RequestException) as error:
                record: dict[str, Any] = {
                    "schema": "opensky-collection-attempt-v1",
                    "attempted_at_utc": attempted_at,
                    "block_id": block_id,
                    "attempt": attempt,
                    "maximum_attempts": config.max_attempts_per_block,
                    "outcome": "rejected",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "bbox": list(config.bbox),
                    "minimum_eligible_states": config.minimum_eligible_states,
                }
                if isinstance(error, EligibilityError):
                    record.update(
                        {
                            "failed_snapshot_index": error.snapshot_index,
                            "eligible_states": error.eligible_states,
                            "eligible_counts_before_rejection": error.eligible_counts,
                        }
                    )
                if attempt_log_path is not None:
                    append_attempt_record(attempt_log_path, record)
                if attempt == config.max_attempts_per_block:
                    raise CollectionError(
                        f"{block_id} failed after {attempt} bounded attempts: {error}"
                    ) from error
                if config.retry_delay_seconds:
                    sleep_fn(config.retry_delay_seconds)

        if block is None:  # Defensive invariant; the retry loop returns or raises.
            raise AssertionError("bounded retry loop ended without a block")
        atomic_write_gzip_json(destination, block)
        read_verified_block(destination)
        if attempt_log_path is not None:
            append_attempt_record(
                attempt_log_path,
                {
                    "schema": "opensky-collection-attempt-v1",
                    "attempted_at_utc": attempted_at,
                    "block_id": block_id,
                    "attempt": block["acceptance_policy"]["accepted_attempt"],
                    "maximum_attempts": config.max_attempts_per_block,
                    "outcome": "accepted",
                    "bbox": list(config.bbox),
                    "minimum_eligible_states": config.minimum_eligible_states,
                    "eligible_counts": [
                        snapshot["eligible_observations"]
                        for snapshot in block["snapshots"]
                    ],
                    "content_sha256": block["integrity"]["content_sha256"],
                },
            )
        completed.append(destination)
        if offset + 1 < count and spacing_seconds:
            sleep_fn(spacing_seconds)
    return completed


def iter_verified_blocks(directory: Path, phase: str) -> Iterable[Path]:
    for path in sorted(directory.glob(f"{phase}-*.json.gz")):
        read_verified_block(path)
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect restartable, pseudonymized OpenSky streaming blocks"
    )
    parser.add_argument("--output", type=Path, default=Path("data/recorded_blocks"))
    parser.add_argument("--phase", choices=["calibration", "formal"], required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--snapshots-per-block", type=int, default=6)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--spacing-seconds", type=float, default=60.0)
    parser.add_argument("--minimum-eligible-states", type=int, default=32)
    parser.add_argument("--max-attempts-per-block", type=int, default=1)
    parser.add_argument("--retry-delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--attempt-log",
        type=Path,
        default=Path("data/results/collection_attempts.jsonl"),
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LAMIN", "LOMIN", "LAMAX", "LOMAX"),
        default=DEFAULT_BBOX,
    )
    args = parser.parse_args()
    config = CollectionConfig(
        phase=args.phase,
        bbox=tuple(args.bbox),
        snapshots_per_block=args.snapshots_per_block,
        interval_seconds=args.interval_seconds,
        minimum_eligible_states=args.minimum_eligible_states,
        max_attempts_per_block=args.max_attempts_per_block,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    paths = collect_blocks(
        args.output,
        args.blocks,
        config,
        args.spacing_seconds,
        args.start_index,
        attempt_log_path=args.attempt_log,
    )
    print(json.dumps({"completed": len(paths), "last": str(paths[-1])}, indent=2))


if __name__ == "__main__":
    main()
