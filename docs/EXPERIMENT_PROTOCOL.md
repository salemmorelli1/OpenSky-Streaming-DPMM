# Frozen Experimental Protocol

## Estimand

The primary estimand for endpoint \(r\) is the population-average paired
architecture contrast across frozen traffic blocks and thinning level \(d\):

\[
\Delta_r(d)=E\{Y_r(\mathrm{SMC},d)-Y_r(\mathrm{VI},d)\}.
\]

No claim is made until the complete analysis set and its provenance manifest
exist.

## Design

The benchmark is a 2 by 3 crossed repeated-measures randomized-block design:

- Architecture: streaming variational DPMM versus collapsed DP-SMC.
- Observation thinning: 0%, 15%, and 30%.
- Blocks: 100 separately scheduled OpenSky short streaming windows.
- Total planned executions: 600.

Each block is recorded once and replayed through all six cells. One deterministic
uniform variate is assigned to every observation within a block. The 30% keep
set is a subset of the 15% keep set, which is a subset of the 0% keep set. This
coupling removes avoidable Monte Carlo variation from thinning contrasts.

## Blindness and leakage control

Model features are log-speed, transformed vertical rate, and heading. A
session-pseudonymous association key may be retained outside the feature matrix
for longitudinal alignment. OpenSky category codes, if present, are held out of
inference and used only by the locked evaluator. Human-readable callsigns and
aircraft identifiers are neither features nor public report fields.

## Outcomes

Primary outcomes are mean prequential log score, synchronized compute latency
per retained observation, and posterior cluster-count stability. Normalized
mutual information against evaluation-only OpenSky categories is secondary
because category coverage is incomplete and the field is not a behavioral
ground truth. SMC particle ESS is an algorithmic health measure; rank-normalized
split R-hat and bulk ESS apply only to genuine independently initialized MCMC
traces used in sensitivity analysis.

## Statistical analysis

For a transformed response \(Y_{madb}\), with method \(m\), dropout level \(d\),
and block \(b\), fit

\[
Y_{mdb}=\beta_0+\beta_A A_m+\beta_L L_d+\beta_Q Q_d+
\beta_{AL}A_mL_d+\beta_{AQ}A_mQ_d+b_{0b}+b_{1b}A_m+\epsilon_{mdb}.
\]

Here \(A\) uses sum coding and \(L,Q\) are orthogonal linear and quadratic
contrasts across 0%, 15%, and 30%. Block-specific random intercepts and, if the
fit is stable, random architecture slopes account for pairing. Inference uses
small-sample degrees-of-freedom correction or a block bootstrap. Endpoint-wise
multiplicity is controlled with Holm adjustment; effect sizes and uncertainty
intervals remain primary.

## Missingness and API outages

Treatment thinning is imposed only after recording. Natural API failure and
field-level missingness are not treatment. Blocks failing the preregistered
minimum-completeness threshold are excluded before method execution and logged
with a reason. Network latency is reported separately from synchronized model
compute time.

Release 1.2 prespecifies at most four complete-block attempts separated by 900
seconds. Every rejected and accepted attempt is written to a local JSONL audit
with its reason, eligible count, box, threshold, and timestamp. The collector
stops after the fourth rejection. It never samples without a ceiling and never
silently discards the attempt history. The 13 release-1.1 formal blocks collected
before this rule was fixed are pilot material and do not enter the amended
formal analysis. See `PROTOCOL_AMENDMENT_2026-09-01.md`.

## Execution lock

Ten calibration blocks are excluded from all formal contrasts. They select the
smallest truncation in the fixed grid whose 95th-percentile residual-component
weight is at most 0.01 and whose boundary-saturation rate is at most 0.05. The
resulting lock records block hashes, \(K\), particle count, rejuvenation window,
seeds, treatment levels, and endpoints. Formal execution requires exactly 100
checksum-verified blocks and the public analyzer requires exactly 600 unique
method-by-thinning rows.
