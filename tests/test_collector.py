import json
import math
import tempfile
import unittest
from pathlib import Path

from opensky_streaming_dpmm.collector import (
    CollectionConfig,
    CollectionError,
    atomic_write_gzip_json,
    build_block,
    canonical_json_bytes,
    collect_blocks,
    read_verified_block,
    sanitize_payload,
    sha256_bytes,
)


def state_row(icao="abc123", speed=210.0, heading=359.0, vertical_rate=2.0):
    row = [None] * 18
    row[0] = icao
    row[1] = "SECRET-CALLSIGN"
    row[2] = "Private country"
    row[3] = 100
    row[4] = 101
    row[5] = 8.2
    row[6] = 50.1
    row[7] = 9000.0
    row[8] = False
    row[9] = speed
    row[10] = heading
    row[11] = vertical_rate
    row[14] = "7700"
    row[16] = 0
    row[17] = 3
    return row


class SequenceClient:
    def __init__(self, eligible_counts):
        self.eligible_counts = iter(eligible_counts)

    def get_states(self, bbox):
        count = next(self.eligible_counts)
        rows = [state_row(icao=f"{index:06x}") for index in range(count)]
        return {"time": 123, "states": rows}, {"network_latency_ms": 1.0}


class CollectorPrivacyTests(unittest.TestCase):
    def test_sanitization_removes_direct_identifiers(self):
        result = sanitize_payload({"time": 123, "states": [state_row()]}, b"x" * 32)
        serialized = canonical_json_bytes(result).decode()
        self.assertNotIn("abc123", serialized)
        self.assertNotIn("SECRET-CALLSIGN", serialized)
        self.assertNotIn("Private country", serialized)
        self.assertNotIn("7700", serialized)
        self.assertEqual(result["eligible_observations"], 1)
        self.assertEqual(len(result["observations"][0]["track_key"]), 24)

    def test_incomplete_model_rows_are_rejected(self):
        result = sanitize_payload(
            {"time": 123, "states": [state_row(speed=None)]}, b"x" * 32
        )
        self.assertEqual(result["eligible_observations"], 0)
        self.assertEqual(result["rejected_incomplete_states"], 1)

    def test_nonfinite_and_malformed_identity_rows_are_rejected(self):
        rows = [
            state_row(speed=math.nan),
            state_row(vertical_rate=math.inf),
            state_row(icao="not-hex"),
        ]
        result = sanitize_payload({"time": 123, "states": rows}, b"x" * 32)
        self.assertEqual(result["eligible_observations"], 0)
        self.assertEqual(result["rejected_incomplete_states"], 3)

    def test_verified_block_detects_changes(self):
        config = CollectionConfig("calibration", (49, 7, 54, 13), 2, 0.0, 2)
        block = build_block(SequenceClient([2, 2]), config, b"x" * 32, "calibration-0001")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.json.gz"
            atomic_write_gzip_json(path, block)
            loaded = read_verified_block(path)
            self.assertEqual(loaded["block_id"], "calibration-0001")

    def test_verified_block_rejects_self_consistent_identity_leak(self):
        config = CollectionConfig("formal", (49, 7, 54, 13), 2, 0.0, 2)
        block = build_block(SequenceClient([2, 2]), config, b"x" * 32, "formal-0001")
        block["snapshots"][0]["observations"][0]["icao24"] = "abc123"
        unsigned = dict(block)
        unsigned["integrity"] = dict(block["integrity"])
        unsigned["integrity"].pop("content_sha256")
        block["integrity"]["content_sha256"] = sha256_bytes(
            canonical_json_bytes(unsigned)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.json.gz"
            atomic_write_gzip_json(path, block)
            with self.assertRaisesRegex(CollectionError, "direct identity"):
                read_verified_block(path)

    def test_collection_config_rejects_non_streaming_block(self):
        with self.assertRaises(ValueError):
            CollectionConfig("formal", (49, 7, 54, 13), 1, 5.0, 10)

    def test_bounded_retry_logs_rejection_and_acceptance(self):
        config = CollectionConfig(
            "formal",
            (49, 7, 54, 13),
            2,
            0.0,
            2,
            max_attempts_per_block=2,
            retry_delay_seconds=3.0,
        )
        client = SequenceClient([1, 2, 2])
        sleeps = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt_log = root / "attempts.jsonl"
            paths = collect_blocks(
                root / "blocks",
                1,
                config,
                0.0,
                client=client,
                salt_path=root / "salt.bin",
                attempt_log_path=attempt_log,
                sleep_fn=sleeps.append,
            )
            block = read_verified_block(paths[0])
            records = [json.loads(line) for line in attempt_log.read_text().splitlines()]

        self.assertEqual(sleeps, [3.0])
        self.assertEqual([record["outcome"] for record in records], ["rejected", "accepted"])
        self.assertEqual(records[0]["eligible_states"], 1)
        self.assertEqual(block["acceptance_policy"]["accepted_attempt"], 2)

    def test_bounded_retry_stops_after_prespecified_limit(self):
        config = CollectionConfig(
            "formal",
            (49, 7, 54, 13),
            2,
            0.0,
            2,
            max_attempts_per_block=2,
        )
        client = SequenceClient([1, 1])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt_log = root / "attempts.jsonl"
            with self.assertRaisesRegex(CollectionError, "failed after 2 bounded attempts"):
                collect_blocks(
                    root / "blocks",
                    1,
                    config,
                    0.0,
                    client=client,
                    salt_path=root / "salt.bin",
                    attempt_log_path=attempt_log,
                    sleep_fn=lambda _: None,
                )
            records = [json.loads(line) for line in attempt_log.read_text().splitlines()]

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["outcome"] == "rejected" for record in records))


if __name__ == "__main__":
    unittest.main()
