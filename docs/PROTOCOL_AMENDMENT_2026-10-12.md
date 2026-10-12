# Release 1.4 endpoint and replay amendment

Date: 2026-10-12 UTC (2026-10-11 in the operator's Eastern time zone).
The public amended benchmark is still 0 of 100 formal blocks and 0 of 600 cells.
This amendment precedes fresh v2 calibration and confirmatory execution.

The design remains two architectures, three nested thinning levels (0, 15, 30%),
ten excluded calibration blocks, and 100 formal blocks with six cells each.
The geographic box, six snapshots, five-second collection spacing, minimum of
32 eligible states, four-attempt limit, and 900-second retry delay remain fixed.
The 13 release-1.1 formal blocks remain excluded pilot material.

## Corrections fixed before calibration

1. Acquisition stops at the first accepted complete-block attempt. Duplicate
   aircraft within a cross-section cannot inflate eligibility. Original API
   times must increase strictly; repeated cross-sections, overlapping accepted
   windows, and incompatible acquisition rules fail validation. OAuth2 refresh
   after a 401 is bounded to one retry.
2. Normal-Gamma component-base means and variances are estimated once from
   transformed observations in the excluded calibration sample. They are frozen
   in the lock and shared by both methods and every thinning level. Formal
   observations never refit this base. Fixed prior shape 2.5, prior mean precision
   0.05, DP concentration 1.5, and heading likelihood concentration 6 are unchanged.
3. The first retained snapshot initializes and trains both methods and is not
   scored. For each later snapshot, every retained aircraft is scored against
   the posterior from strictly earlier snapshots, before any current-snapshot
   allocation or update. This is a mean of marginal pre-snapshot log densities,
   not a joint batch log density. SMC's observation-by-observation evidence
   increments remain internal bookkeeping and are not this comparison endpoint.
4. Occupied clusters means the number of distinct labels in the final fitted
   partition: final-posterior maximum responsibility for VI and the highest-weight
   particle's allocations for SMC. NMI uses these same partitions. A highest-weight
   particle is a representative draw, not a certified global MAP partition.
5. Category NMI is secondary. Codes 0 and 1 carry no evaluator information; code
   13 is reserved. These are excluded. NMI is missing below two usable category
   observations, and each cell and aggregate report includes coverage counts.
6. SMC health ESS is recorded before resampling. A uniform reset cannot replace
   the degeneracy measurement that triggered resampling. MCMC rank diagnostics
   remain separate; nonfinite draws and traces with no estimable diagnostic
   variance are rejected. The bulk-ESS implementation conservatively caps ESS
   at the number of split draws; it is not tail ESS or a mixing proof.

## Calibration and timing

The grid remains K = 8, 12, 16, 24, 32. Each calibration block is replayed at
all three thinning levels. Its residual mass is the worst final residual across
those levels; its saturation indicator is one if any level occupies all K
components at the existing 0.01 posterior-weight threshold. The smallest usable
K with residual q95 <= 0.01 and saturation rate <= 0.05 is selected. Only the
initial retained snapshot requires at least K observations; later snapshots
must be nonempty. An unusable candidate has null diagnostics, not JSON Infinity.
Failure of the fixed grid stops confirmatory calibration. Expanding the grid or
changing model settings requires a separately documented amendment and code
review; validation-only experiments cannot change the public design.

Both methods are warmed before timing. Timing covers model initialization,
pre-snapshot predictive queries, updates, and final partition extraction.
Acquisition, parsing, feature transfer, thinning, and category NMI are outside
it. Divide by all retained observations, including the unscored training
snapshot. The lock freezes code, CPU/device description, package versions,
precision, and thread settings. It must be generated in the same compute
environment used for replay.

## Evidence and publication

Use isolated `data/recorded_blocks/amended-v2` and `data/results/amended-v2`
paths. Preserve older blocks, locks, and results separately. The v2 loader
rejects v1 locks; no earlier checkpoint can silently become v2 evidence. A new
lock is never written over an existing lock.

Formal windows must follow calibration and must not overlap. Every checkpoint
row binds its lock, implementation, environment, block hash, endpoint protocol,
counts, and metrics through a canonical typed row fingerprint. Analysis rereads
original verified blocks and checks deterministic mask counts, category
coverage, compatible health metrics, and latency arithmetic. Public publication
still requires exactly 100 formal blocks and 600 successful unique finite cells.
A programmatic smaller experiment must opt into validation-only scope, which
never updates public completion status. Missing secondary results serialize
as JSON null. The website verifies summary bytes and matching status provenance,
then requires complete primary contrasts before showing results.

Hashes detect inconsistent or accidentally modified artifacts. They do not
cryptographically authenticate an OpenSky server or independently prove an
execution. Paired t inference still assumes sufficiently independent blocks;
source timestamps do not establish that assumption. Collection scheduling,
acceptance histories, temporal dependence, and any deviations must be disclosed.
The committed release verifier deliberately audits the current methods-only,
0/600 release; a future empirical publication needs its own reviewed artifact
and status update. No live acquisition or empirical outcome was added by this
software audit.
