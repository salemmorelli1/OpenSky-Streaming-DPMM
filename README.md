# OpenSky Streaming DPMM

[![Validate](https://github.com/salemmorelli1/OpenSky-Streaming-DPMM/actions/workflows/validate.yml/badge.svg)](https://github.com/salemmorelli1/OpenSky-Streaming-DPMM/actions/workflows/validate.yml)
[![Pages](https://github.com/salemmorelli1/OpenSky-Streaming-DPMM/actions/workflows/pages.yml/badge.svg)](https://github.com/salemmorelli1/OpenSky-Streaming-DPMM/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](LICENSE)

A topology-aware, open-world Bayesian clustering laboratory for streaming
aircraft kinematics from the OpenSky Network. The project compares a truncated
streaming variational Dirichlet-process mixture with a collapsed DP sequential
Monte Carlo method under controlled observation thinning.

**Release 1.1 status:** authenticated live-API connectivity has been verified on
the local research machine, and the repository now contains the complete
restartable acquisition, calibration, six-cell replay, and aggregate-analysis
pipeline. The 100-block/600-execution formal benchmark remains pending; no
comparative performance result is fabricated or inferred from the connectivity
smoke test.

## Research question

Can an online nonparametric mixture preserve stable, calibrated latent traffic
structure when observations arrive as a stream, headings live on a circle, the
number of behavioral clusters is unknown, and 0%-30% of updates are removed?

## Statistical correction

The observation space is

\[
\mathcal Y=\mathbb R^2\times S^1,
\]

with reference measure equal to two-dimensional Lebesgue measure times arc-
length Haar measure. The component kernel is the product of a Euclidean
Normal-Gamma predictive distribution and a von Mises directional density. This
avoids the singular-measure error created by embedding heading as
\((\cos\theta,\sin\theta)\) and then assigning a nonsingular four-dimensional
Gaussian density.

The variational approximation uses exactly \(K-1\) beta-distributed sticks. The
\(K\)th component carries the residual mass, so expected weights sum to one.
Cumulative inference and exponential forgetting are exposed as different
targets; constant-gain tracking is never described as ordinary posterior
consistency.

## Methods

| Method | Inferential role | Geometry | Online mechanism |
|---|---|---|---|
| Streaming variational DPMM | Low-latency approximation | Normal-Gamma x von Mises | Cumulative or power-forgetting sufficient statistics |
| Collapsed DP-SMC | Posterior comparator | Collapsed Euclidean/circular predictive | Fully adapted allocation, systematic resampling, Gibbs rejuvenation |

## Design

The frozen benchmark is a 2 architecture x 3 thinning-level crossed repeated-
measures randomized-block experiment across 100 recorded blocks: 600 planned
executions. Every method receives the same block and nested deterministic masks.
Natural API missingness is recorded separately from treatment thinning.

| Factor | Levels |
|---|---|
| Architecture | Variational DPMM; collapsed DP-SMC |
| Controlled thinning | 0%; 15%; 30% |
| Block | 100 frozen OpenSky windows |

## Quick start

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
python -m pip install --upgrade pip
python -m pip install -e ".[test,report]"

python -m opensky_streaming_dpmm.engine
python -m pytest
```

Run one descriptive live snapshot:

```bash
python -m opensky_streaming_dpmm.engine --live
```

For authenticated access, export `OPENSKY_CLIENT_ID` and
`OPENSKY_CLIENT_SECRET`. Credentials are read from the environment and must
never be committed.

Rebuild publication artifacts:

```bash
python scripts/build_site.py
python scripts/build_report.py
```

## Execute the empirical benchmark

Raw blocks and credentials remain local. The collector discards callsign,
origin-country, squawk, and raw ICAO24 before serialization; the retained track
key is a keyed HMAC using a local, Git-ignored salt. Every compressed block has a
content checksum and hashes of the original API responses.

After loading `.env.local`, collect the ten calibration blocks:

```bash
python -m opensky_streaming_dpmm.collector \
  --phase calibration --blocks 10 \
  --snapshots-per-block 6 --interval-seconds 5 \
  --spacing-seconds 900 --minimum-eligible-states 32

unset OPENSKY_CLIENT_ID OPENSKY_CLIENT_SECRET
python -m opensky_streaming_dpmm.empirical calibrate
```

Then collect 100 separately scheduled formal blocks, execute the paired design,
and unlock the aggregate analysis:

```bash
source .env.local
python -m opensky_streaming_dpmm.collector \
  --phase formal --blocks 100 \
  --snapshots-per-block 6 --interval-seconds 5 \
  --spacing-seconds 900 --minimum-eligible-states 32

unset OPENSKY_CLIENT_ID OPENSKY_CLIENT_SECRET
python -m opensky_streaming_dpmm.empirical run
python -m opensky_streaming_dpmm.empirical analyze
python scripts/build_site.py
```

For publication-quality temporal coverage, collect the formal set as ten
10-block sessions across multiple days rather than one uninterrupted session.
See the [empirical execution guide](docs/EMPIRICAL_EXECUTION_GUIDE.md) for exact
restart and start-index commands.

## Repository map

```text
src/opensky_streaming_dpmm/engine.py      Core inference algorithms
src/opensky_streaming_dpmm/collector.py   Restartable privacy-minimized acquisition
src/opensky_streaming_dpmm/empirical.py   Calibration, 600-cell replay, analysis gate
tests/                                  Numerical, privacy, and completeness invariants
docs/STATISTICAL_MODEL.md              Full mathematical reconstruction
docs/EXPERIMENT_PROTOCOL.md            Frozen 2 x 3 analysis plan
docs/EMPIRICAL_EXECUTION_GUIDE.md       Exact local execution sequence
docs/CLAIM_BOUNDARY.md                 Supported and unsupported conclusions
scripts/build_site.py                  Reproducible interactive GitHub Pages site
scripts/build_report.py                Reproducible 26-page APA-style report
report/                                Report PDF and figures
data/                                  Design and machine-readable status
```

## Evidence and claims

OpenSky supplies aircraft state vectors rather than received radar pulses. This
repository is not evidence of radar-pulse classification, emitter
identification, ELINT/SIGINT performance, or operational validation. Category
metadata is reserved for post-hoc evaluation and never enters model fitting.

## Data responsibility

The repository does not redistribute OpenSky state-vector records. Users are
responsible for complying with the OpenSky API terms, attribution requirements,
rate limits, privacy expectations, and local law. Recorded evaluation blocks
belong under ignored `data/recorded_blocks/`; aggregate, non-identifying results
may be committed after the frozen benchmark is executed.

## Documentation and live laboratory

- [Interactive research laboratory](https://salemmorelli1.github.io/OpenSky-Streaming-DPMM/)
- [26-page APA-style methods report](report/OpenSky_Streaming_DPMM_APA_Report.pdf)
- [Statistical model](docs/STATISTICAL_MODEL.md)
- [Experimental protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Claim boundary](docs/CLAIM_BOUNDARY.md)
- [Empirical execution guide](docs/EMPIRICAL_EXECUTION_GUIDE.md)

## References

- Blei, D. M., and Jordan, M. I. (2006). Variational inference for Dirichlet
  process mixtures. *Bayesian Analysis, 1*(1), 121-144.
- Sethuraman, J. (1994). A constructive definition of Dirichlet priors.
  *Statistica Sinica, 4*, 639-650.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Buerkner, P. C. (2021).
  Rank-normalization, folding, and localization: An improved R-hat for assessing
  convergence of MCMC. *Bayesian Analysis, 16*(2), 667-718.
- OpenSky Network. (2026). *REST API documentation*.
  https://openskynetwork.github.io/opensky-api/rest.html

## License

Software is released under the [MIT License](LICENSE). OpenSky data remain
subject to their source terms and are not relicensed by this repository.
