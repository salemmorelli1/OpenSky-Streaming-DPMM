import copy
import json
from unittest.mock import Mock

import pytest
import requests

from opensky_streaming_dpmm.collector import (
    CollectionConfig,
    CollectionError,
    OAuthSession,
    atomic_write_gzip_json,
    build_block,
    canonical_json_bytes,
    collect_blocks,
    read_verified_block,
    sanitize_payload,
    sha256_bytes,
)
from opensky_streaming_dpmm.engine import OpenSkyClient
from test_collector import SequenceClient, state_row


def rehash(block):
    unsigned = copy.deepcopy(block)
    unsigned["integrity"].pop("content_sha256", None)
    block["integrity"]["content_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    return block


def test_first_success_stops_bounded_retry(tmp_path):
    config = CollectionConfig("formal", (49, 7, 54, 13), 2, 0, 2, 4, 900)
    client = SequenceClient([2, 2])
    sleeps = []
    paths = collect_blocks(
        tmp_path / "blocks",
        1,
        config,
        0,
        client=client,
        salt_path=tmp_path / "salt",
        attempt_log_path=tmp_path / "attempts.jsonl",
        sleep_fn=sleeps.append,
    )
    assert client.calls == 2
    assert sleeps == []
    assert read_verified_block(paths[0])["acceptance_policy"]["accepted_attempt"] == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "attempts.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1 and records[0]["outcome"] == "accepted"


def test_early_success_after_failure_is_not_replaced(tmp_path):
    config = CollectionConfig("formal", (49, 7, 54, 13), 2, 0, 2, 4, 3)
    client = SequenceClient([1, 2, 2])
    sleeps = []
    paths = collect_blocks(
        tmp_path / "blocks",
        1,
        config,
        0,
        client=client,
        salt_path=tmp_path / "salt",
        sleep_fn=sleeps.append,
    )
    assert client.calls == 3 and sleeps == [3]
    assert read_verified_block(paths[0])["acceptance_policy"]["accepted_attempt"] == 2


def test_resume_checks_acquisition_rule_before_fetch(tmp_path):
    config = CollectionConfig("formal", (49, 7, 54, 13), 2, 0, 2)
    client = SequenceClient([2, 2])
    collect_blocks(
        tmp_path / "blocks", 1, config, 0, client=client, salt_path=tmp_path / "salt"
    )
    changed = CollectionConfig("formal", (49, 7, 54, 13), 2, 0, 3)
    with pytest.raises(CollectionError, match="different acquisition protocol"):
        collect_blocks(
            tmp_path / "blocks",
            1,
            changed,
            0,
            client=Mock(),
            salt_path=tmp_path / "salt",
        )


def test_duplicate_aircraft_do_not_inflate_eligibility():
    row = state_row()
    payload = sanitize_payload({"time": 123, "states": [row, row, None, 42]}, b"x" * 32)
    assert payload["eligible_observations"] == 1
    assert payload["rejected_incomplete_states"] == 3


@pytest.mark.parametrize("time", [None, True, "123", 1.5, 0, -1])
def test_source_timestamp_must_be_present_and_integral(time):
    with pytest.raises(CollectionError, match="response time"):
        sanitize_payload({"time": time, "states": []}, b"x" * 32)


def test_repeated_api_snapshot_is_rejected():
    client = Mock()
    client.get_states.return_value = (
        {"time": 123, "states": [state_row(), state_row("abc124")]},
        {},
    )
    with pytest.raises(CollectionError, match="timestamps must increase"):
        build_block(
            client,
            CollectionConfig("formal", (49, 7, 54, 13), 2, 0, 2),
            b"x" * 32,
            "formal-0001",
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda b: b["snapshots"][1].update(api_time=b["snapshots"][0]["api_time"]),
            "timestamps",
        ),
        (
            lambda b: b["snapshots"][0]["observations"].append(
                b["snapshots"][0]["observations"][0].copy()
            ),
            "eligible count",
        ),
        (
            lambda b: b["snapshots"][0]["transport"].update(icao24="abc123"),
            "direct identity",
        ),
        (
            lambda b: b["acceptance_policy"].update(accepted_attempt=5),
            "acquisition policy",
        ),
        (
            lambda b: b["snapshots"][0]["observations"][0].update(category=99),
            "invalid category",
        ),
    ],
)
def test_checksum_alone_does_not_accept_malformed_block(tmp_path, mutation, match):
    config = CollectionConfig("formal", (49, 7, 54, 13), 2, 0, 2)
    block = build_block(SequenceClient([2, 2]), config, b"x" * 32, "formal-0001")
    mutation(block)
    path = tmp_path / "block.json.gz"
    atomic_write_gzip_json(path, rehash(block))
    with pytest.raises(CollectionError, match=match):
        read_verified_block(path)


def test_engine_and_collector_use_same_sanitized_observations():
    payload = {
        "time": 123,
        "states": [state_row(), state_row(speed=float("nan")), state_row(icao="bad")],
    }
    client = OpenSkyClient(pseudonym_salt=b"x" * 32)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            client._transport,
            "get_states",
            lambda bbox: (payload, {"network_latency_ms": 1}),
        )
        snapshot = client.fetch()
    sanitized = sanitize_payload(payload, b"x" * 32)
    assert snapshot.euclidean.shape == (1, 2)
    assert snapshot.track_keys == tuple(
        r["track_key"] for r in sanitized["observations"]
    )


def response(status, payload=None):
    result = Mock(status_code=status, headers={})
    result.json.return_value = payload
    result.raise_for_status.side_effect = (
        requests.HTTPError(str(status)) if status >= 400 else None
    )
    return result


def test_oauth_refreshes_revoked_token_only_once():
    session = Mock()
    session.post.side_effect = [
        response(200, {"access_token": "first", "expires_in": 1800}),
        response(200, {"access_token": "second", "expires_in": 1800}),
    ]
    session.get.side_effect = [
        response(401),
        response(200, {"time": 123, "states": []}),
    ]
    payload, _ = OAuthSession("test-id", "test-secret", session=session).get_states(
        (49, 7, 54, 13)
    )
    assert payload["time"] == 123
    assert session.post.call_count == session.get.call_count == 2
    assert session.get.call_args_list[-1].kwargs["headers"] == {
        "Authorization": "Bearer second"
    }


def test_oauth_does_not_loop_after_second_unauthorized_response():
    session = Mock()
    session.post.return_value = response(
        200, {"access_token": "test", "expires_in": 1800}
    )
    session.get.return_value = response(401)
    with pytest.raises(requests.HTTPError):
        OAuthSession("test-id", "test-secret", session=session).get_states(
            (49, 7, 54, 13)
        )
    assert session.get.call_count == 2
