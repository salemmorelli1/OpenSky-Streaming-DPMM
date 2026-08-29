# Claim Boundary

This repository implements and tests a topology-aware Bayesian clustering
methodology for public OpenSky aircraft state vectors. It does not ingest raw
radio-frequency measurements, radar pulses, pulse descriptor words, modulation
features, or intercepted emitter waveforms. Consequently, it does not validate
radar deinterleaving, emitter identification, ELINT, SIGINT, or operational
surveillance performance.

## Supported at release 1.0

- A product-space likelihood on Euclidean kinematics and circular heading.
- Correct residual-stick truncation for a variational Dirichlet-process mixture.
- A collapsed DP sequential Monte Carlo comparator with resampling and Gibbs
  rejuvenation.
- Deterministic nested thinning masks and a preregistered 2 by 3 repeated-
  measures randomized-block design.
- Current OpenSky state-vector ingestion with optional OAuth2 credentials.
- Software tests, reproducible site/report generation, and explicit provenance.

## Supported only after the frozen benchmark is executed

- Comparative clustering quality, latency, cluster-count stability, calibration,
  or robustness under thinning.
- Population-average method contrasts and method-by-thinning interactions.
- Any numerical endpoint or confidence interval derived from the 600 planned
  executions.

## Not supported by this project

- Claims about aircraft identity, threat status, intent, or sensitive operations.
- Radar pulse-train classification or hardware receiver performance.
- Operational or fielded decision-support validation.

The project should be described as an aviation-surveillance methodology study
using public kinematic state vectors.
