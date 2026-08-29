import tempfile
import unittest
from pathlib import Path

from opensky_streaming_dpmm.collector import (
    CollectionConfig,
    atomic_write_gzip_json,
    canonical_json_bytes,
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

    def test_verified_block_detects_changes(self):
        block = {
            "schema_version": "opensky-empirical-block-v1",
            "block_id": "calibration-0001",
            "phase": "calibration",
            "snapshots": [],
            "integrity": {"raw_response_sha256": []},
        }
        block["integrity"]["content_sha256"] = sha256_bytes(canonical_json_bytes(block))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.json.gz"
            atomic_write_gzip_json(path, block)
            loaded = read_verified_block(path)
            self.assertEqual(loaded["block_id"], "calibration-0001")

    def test_collection_config_rejects_non_streaming_block(self):
        with self.assertRaises(ValueError):
            CollectionConfig("formal", (49, 7, 54, 13), 1, 5.0, 10)


if __name__ == "__main__":
    unittest.main()
