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
- Blocks: 100 independently recorded OpenSky snapshots or short windows.
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

Primary outcomes are adjusted mutual information or normalized mutual
information against evaluation-only categories when label coverage is adequate,
prequential log score, synchronized compute latency, and posterior cluster-count
stability. SMC particle ESS is an algorithmic health measure; rank-normalized
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
