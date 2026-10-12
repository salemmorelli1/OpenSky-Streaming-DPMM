# OpenSky replay integrity audit

Audited base: `022af31f118e7cb594029d6ac8532d481b17f197`.
Audit date: 2026-10-12 UTC, corresponding to 2026-10-11 Eastern time.
Scope: current source, statistical targets, acquisition, calibration, replay,
checkpointing, aggregate publication, packaging, CI, site, and methods report.
Prior merged release-1.3 controls and the bounded-retry amendment were retained.
The baseline's 25 tests passed before changes; they did not cover the defects
below. The corrected built wheel passes 109 tests locally on Python 3.12 with
PyTorch 2.14.1+cpu and the committed dependency lock.

| Finding | Correction | Verification |
|---|---|---|
| A successful acquisition did not break the retry loop, so later attempts could replace or discard accepted data. | Stop immediately at first complete success; preserve its accepted-attempt metadata. | First-attempt and second-attempt success tests verify exact request counts and audit outcomes. |
| VI scored a whole snapshot from its previous posterior, but SMC's endpoint incorporated earlier aircraft from the same snapshot. | Read-only frozen-posterior SMC prediction; both architectures score before every current-snapshot update. | Analytic CRP/particle-mixture comparison, order/RNG/state invariance, and runner-level score checks. |
| Component-base moments were refitted on each formal cell's initial snapshot. | Freeze shared base moments from excluded calibration data. | Both architectures keep identical base parameters across different initial samples. |
| VI counted components above a weight threshold while SMC counted labels; their NMI allocations also referred to different fit stages. | Use final representative partitions and distinct labels for both occupancy endpoints and NMI. | Allocation/count equality and evaluator-label blindness tests. |
| Duplicate aircraft, repeated API times, and inconsistent resume protocols could pass checksums. | Enforce unique per-snapshot keys, strict chronology, nonoverlapping windows, and acquisition-rule consistency. | Rehashed malformed-block controls and incompatible resume/window tests. |
| Locks and checkpoint rows did not bind code, compute environment, mask counts, or finite edits to CSV endpoints. | v2 lock, fixed confirmatory settings, typed row fingerprints, count/latency/coverage checks, and original-block verification during analysis. | Changed implementation/environment, stale blocks, finite metric edits, duplicate/out-of-design rows, and altered model settings fail closed. |
| The expected-block override could mark a tiny experiment as a completed empirical benchmark. | Explicit validation-only scope; only 100 blocks and 600 cells can publish. | A real 12-cell synthetic pipeline stays private; a temporary synthetic 600-row control exercises the full gate without changing repository status. |
| SMC ESS after resampling hid degeneracy; invalid diagnostic traces could return misleading results. | Record pre-resampling ESS and reject nonfinite/constant diagnostic input. | Forced resampling, independent/AR(1) traces, and invalid-draw tests. |
| The site requested contrast key `0` while Python emitted `0.0`; small/stale summaries could be displayed; missing values could emit nonstandard JSON NaN. | Correct key; full status/provenance/byte-fingerprint gate; explicit JSON null for missing summaries. | Node executes the actual generated JavaScript against real analyzer output and rejects incomplete/stale evidence. |
| Dependencies were unconstrained, Windows was untested, and report prose disagreed with the code's directional update and one-pass VI behavior. | Pinned CPU lock; Linux/Windows CI; wheel checks; revised and rendered methods report; source/report/site release verification. | Local wheel, lint, compile, CLI/engine smoke, PDF text/page checks, and visual inspection of all 26 rendered pages. |

The main corrected Python files are `engine.py`, `collector.py`, `empirical.py`,
and the new lightweight `contracts.py`, under `src/opensky_streaming_dpmm`.
The package initializer, both publication builders, the new `verify_release.py`,
and the regression tests are included in the tested revision.

The end-to-end test uses synthetic state vectors in temporary directories. It
writes verified gzip blocks, actually calibrates, executes all 12 paired cells,
resumes without recomputation, and analyzes privately. A separate 600-row gate
fixture is also synthetic and temporary. Neither is an OpenSky benchmark,
performance finding, or substitute for the 100-block study. No credentials,
recorded operator data, or live API calls were used for this audit. The earlier
operator connectivity check retains its original 2026-08-29 date and scope.

The corrected release requires fresh v2 calibration. The design remains six
cells per block, 100 formal blocks, 600 executions, and ten excluded calibration
blocks. Its public status remains 0/600. See the dated protocol amendment for
precise endpoint, timing, and calibration definitions. GPU execution, live
service availability, population independence, category validity, full posterior
cluster-count uncertainty, and real-time feasibility are not established by
these CPU software tests. SMC still retains transformed histories and its memory
grows with block length and particle count.

GitHub verification is performed on the feature head and PR merge revision.
The exact head/tree, successful workflow links, and downloadable tested archive
are recorded in the PR and delivery. Deployment workflow and main are not
changed by this audit publication.

Primary contracts checked:

- OpenSky's current REST schema, OAuth2 refresh, and rate-limit headers:
  https://openskynetwork.github.io/opensky-api/rest.html
- Rank diagnostics and ESS definitions:
  https://mc-stan.org/docs/reference-manual/analysis.html
- Variational DP mixture reference used by the existing model:
  https://doi.org/10.1214/06-BA104

Checksums are integrity evidence, not signatures of API origin or proof that
arbitrary CSV rows were executed. Formal outcome interpretation requires the
operator's acquisition audit, frozen blocks, and prespecified temporal coverage.
