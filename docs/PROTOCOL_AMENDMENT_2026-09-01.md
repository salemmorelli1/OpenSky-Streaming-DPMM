# Protocol Amendment: Bounded Acquisition Retry

## Date and trigger

This amendment was fixed on 2026-09-01 after the release-1.1 collector repeatedly
encountered formal-block first snapshots below the minimum of 32 eligible model
observations. Thirteen checksum-valid formal blocks had been collected before the
amendment. Those blocks remain preserved as pilot material and are excluded from
the amended confirmatory experiment.

## Decision

The WGS84 box remains `(49, 7, 54, 13)`, the block remains six snapshots separated
by five seconds, and the minimum remains 32 eligible observations in every
snapshot. Changing the box was rejected because it would change the target
population. Retrying without a limit was rejected because it would conceal the
sampling mechanism and condition collection on an unspecified stopping rule.

For every amended calibration or formal block, the collector may make at most
four complete-block attempts separated by 900 seconds. Any snapshot below 32
rejects its complete attempt. A connection or rate-limit failure also rejects the
attempt. After the fourth rejection, acquisition stops and requires a separately
scheduled session. No partial attempt is serialized as a block.

## Audit and data separation

Every attempt is appended to a local JSONL audit containing its UTC timestamp,
block identifier, attempt number, outcome, box, threshold, and failure reason or
accepted eligible counts. It contains no callsign, raw ICAO24 value, or OAuth
credential. Accepted blocks record the retry policy and accepted attempt number
inside the checksum-protected artifact.

Amended blocks are written beneath `data/recorded_blocks/amended`; the lock and
row-level results are written beneath `data/results/amended`. Both parents are
Git-ignored. The release-1.1 blocks are not deleted, renumbered, or mixed into the
amended analysis.

## Consequences

Calibration and formal numbering restart at one under the amended rule. A new
calibration lock must be generated before collecting the new 100 formal blocks.
The 600-cell analysis gate is unchanged. The final report must disclose this
amendment, the 13 excluded pilot blocks, the acceptance rate, and all exhausted
four-attempt sessions.
