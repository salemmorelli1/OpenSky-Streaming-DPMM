# Empirical OpenSky Execution Guide

## Scope

Release 1.2 adds the bounded-retry amendment documented in
`PROTOCOL_AMENDMENT_2026-09-01.md`. The authenticated one-snapshot command and
the 13 release-1.1 formal blocks are pilot evidence only. The amended formal
experiment begins in the separate `data/recorded_blocks/amended` directory.

The raw pseudonymized blocks, local pseudonym key, locked configuration, and
row-level results remain ignored by Git. Only the completed aggregate summary
may be published after the 600-cell completeness gate succeeds.

## Fixed acquisition unit

One block contains six OpenSky `/api/states/all` cross-sections separated by five
seconds, giving a short streaming window rather than a static snapshot. A block
is rejected before inference if any cross-section contains fewer than 32 complete
model observations. Network latency and rate-limit headers are recorded, but
network time is excluded from compute latency.

The default WGS84 box is `(49, 7, 54, 13)`. Changing the box changes the target
population and therefore requires a new calibration and analysis lock.

Each amended block has at most four complete-block attempts separated by 900
seconds. A failed snapshot rejects the entire attempt. All attempts are logged
to `data/results/amended/collection_attempts.jsonl`; after four failures the
collector stops. This is a bounded acceptance rule, not sampling until success.

## 1. Load local credentials

From Windows Git Bash:

```bash
cd "/c/Users/salem/GitHub/OpenSky-Streaming-DPMM"
source .venv/Scripts/activate
source .env.local
```

The repository never reads `credentials.json`, and `.env.local` is excluded from
Git. Do not paste a secret into source code, workflow YAML, an issue, or a commit.

## 2. Collect the isolated calibration sample

Collect ten calibration blocks at separately scheduled times:

```bash
python -m opensky_streaming_dpmm.collector \
  --output data/recorded_blocks/amended \
  --phase calibration \
  --blocks 10 \
  --snapshots-per-block 6 \
  --interval-seconds 5 \
  --spacing-seconds 900 \
  --minimum-eligible-states 32 \
  --max-attempts-per-block 4 \
  --retry-delay-seconds 900 \
  --attempt-log data/results/amended/collection_attempts.jsonl
```

The command is restartable. Existing blocks are checksum-verified and skipped.
For stronger temporal coverage, collect smaller batches across multiple days by
using `--blocks` with the next `--start-index`.

## 3. Freeze the truncation and compute configuration

```bash
unset OPENSKY_CLIENT_ID
unset OPENSKY_CLIENT_SECRET
python -m opensky_streaming_dpmm.empirical calibrate \
  --blocks data/recorded_blocks/amended \
  --output data/results/amended/locked_config.json
```

Calibration evaluates `K = 8, 12, 16, 24, 32`. The smallest value is selected
only if the 95th percentile of the final residual-component weight is at most
0.01 and the boundary-saturation rate is at most 0.05. Failure is a hard stop:
expand the candidate grid, document the amendment, and recalibrate.

The output `data/results/amended/locked_config.json` contains the calibration
block checksums, selected truncation, particle count, seeds, and endpoints.

## 4. Collect 100 formal blocks

Reconnect the local environment. Ten separately scheduled sessions of ten
blocks give better temporal coverage than one uninterrupted burst:

```bash
source .env.local
python -m opensky_streaming_dpmm.collector \
  --output data/recorded_blocks/amended \
  --phase formal \
  --blocks 10 \
  --start-index 1 \
  --snapshots-per-block 6 \
  --interval-seconds 5 \
  --spacing-seconds 900 \
  --minimum-eligible-states 32 \
  --max-attempts-per-block 4 \
  --retry-delay-seconds 900 \
  --attempt-log data/results/amended/collection_attempts.jsonl
```

Repeat with start indices `11, 21, ..., 91`. Vary collection times across days.
The formal runner rejects calibration blocks, duplicate identifiers, checksum
failures, and any block count other than exactly 100.

## 5. Execute all paired cells

```bash
unset OPENSKY_CLIENT_ID
unset OPENSKY_CLIENT_SECRET
python -m opensky_streaming_dpmm.empirical run \
  --blocks data/recorded_blocks/amended \
  --lock data/results/amended/locked_config.json \
  --output data/results/amended/factorial_results.csv
```

Each block is replayed through both methods and nested 0%, 15%, and 30%
thinning masks. Cell order is randomized within block from the locked seed. The
CSV is checkpointed after every new cell, so an interrupted CPU run resumes
without recomputing finished cells.

## 6. Unlock the aggregate analysis

```bash
python -m opensky_streaming_dpmm.empirical analyze \
  --results data/results/amended/factorial_results.csv
python scripts/build_site.py
python -m pytest -q
```

The analyzer refuses to run unless it finds exactly 100 blocks and 600 unique
cells. On success it creates `data/empirical_summary.json` and changes
`data/project_status.json` to `empirical_benchmark_complete`. Commit only those
aggregate artifacts and the rebuilt page—not raw blocks, secrets, the lock file,
or row-level results.

## Interpretation boundary

The completed experiment supports conclusions about topology-aware clustering
of OpenSky state vectors under controlled observation thinning. It does not
validate received radar pulses, emitter identity, intent, ELINT/SIGINT
performance, or operational deployment.
