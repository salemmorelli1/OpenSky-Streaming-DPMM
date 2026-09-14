# Release 1.3.0

- Recomputes and verifies the frozen analysis-lock fingerprint before formal
  replay and public analysis.
- Binds every restart checkpoint row to one verified lock and block checksum;
  duplicate, stale, unsuccessful, and out-of-design rows fail closed.
- Requires ten unique calibration blocks and checks truncation usability under
  the maximum prespecified thinning level.
- Requires finite primary endpoints before the 600-cell publication gate can
  unlock, and adds occupied-cluster contrasts and endpoint-wise Holm-adjusted
  p-values.
- Publishes hashes of the lock, factorial result file, and formal-block manifest
  in the aggregate summary.
- Rejects non-finite model features, malformed ICAO24 values, identity leaks,
  malformed snapshot structure, and inconsistent block manifests.
- Treats OpenSky category codes 0 and 1 as missing evaluator labels.
- Makes the generated PDF deterministic and makes CI reject stale generated
  publication artifacts.

The empirical benchmark remains pending. This release changes validation and
the prespecified block-contrast implementation; it does not publish or imply an
OpenSky performance result.
