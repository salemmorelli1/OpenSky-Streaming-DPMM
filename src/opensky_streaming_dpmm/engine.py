"""Topology-aware streaming Bayesian mixture baseline for OpenSky state vectors.

This module deliberately separates four objects that the original draft mixed:
API ingestion, blinded model features, evaluation-only labels, and MCMC
diagnostics.  It implements a truncated stick-breaking streaming variational
mixture and a collapsed DP particle filter with Gibbs rejuvenation.  It does not
claim that OpenSky kinematics are radar pulses or that a live pull is a
reproducible experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import requests
import torch
from scipy.stats import norm, rankdata

from .collector import CollectionError, OAuthSession, pseudonymize, sanitize_payload
from .contracts import finite_real, integer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
LOG_2PI = math.log(2.0 * math.pi)


class OpenSkyError(RuntimeError):
    """Raised when the OpenSky service returns an unusable response."""


@dataclass(frozen=True)
class OpenSkySnapshot:
    """One timestamped cross-section with model and evaluation data separated."""

    time_position: int
    euclidean: torch.Tensor  # [log(1 + speed), asinh(vertical_rate / 5)]
    heading: torch.Tensor  # radians on S^1
    track_keys: tuple[str, ...]  # session-pseudonymous; never a model feature
    categories: np.ndarray  # evaluation only; -1 denotes missing
    network_latency_ms: float

    def model_view(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.euclidean, self.heading


class OpenSkyClient:
    """Current OpenSky /api/states/all client with optional OAuth2 credentials."""

    API_URL = "https://opensky-network.org/api/states/all"
    TOKEN_URL = (
        "https://auth.opensky-network.org/auth/realms/opensky-network/"
        "protocol/openid-connect/token"
    )

    def __init__(
        self,
        bbox: tuple[float, float, float, float] = (49.0, 7.0, 54.0, 13.0),
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout_seconds: float = 10.0,
        pseudonym_salt: bytes | None = None,
    ) -> None:
        lamin, lomin, lamax, lomax = bbox
        if not (-90 <= lamin < lamax <= 90 and -180 <= lomin < lomax <= 180):
            raise ValueError("bbox must be (lamin, lomin, lamax, lomax) in WGS84")
        if (client_id is None) != (client_secret is None):
            raise ValueError("provide both OAuth2 client_id and client_secret")
        finite_real(timeout_seconds, "timeout_seconds", positive=True)

        self.params = {
            "lamin": lamin,
            "lomin": lomin,
            "lamax": lamax,
            "lomax": lomax,
            "extended": 1,
        }
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout_seconds = timeout_seconds
        self.salt = pseudonym_salt or os.urandom(32)
        self.session = requests.Session()
        self._transport = OAuthSession(
            client_id, client_secret, timeout_seconds, self.session
        )

    def _headers(self) -> dict[str, str]:
        return self._transport._headers()

    def _pseudonym(self, icao24: str) -> str:
        return pseudonymize(icao24.lower(), self.salt)

    def fetch(self) -> OpenSkySnapshot:
        self._transport.session = self.session
        bbox = tuple(self.params[key] for key in ("lamin", "lomin", "lamax", "lomax"))
        try:
            payload, metadata = self._transport.get_states(bbox)
            sanitized = sanitize_payload(payload, self.salt)
        except CollectionError as error:
            raise OpenSkyError(str(error)) from error
        observations = sanitized["observations"]
        euclidean = [
            [math.log1p(row["velocity"]), math.asinh(row["vertical_rate"] / 5.0)]
            for row in observations
        ]
        headings = [math.radians(row["true_track"]) for row in observations]

        x = torch.as_tensor(euclidean, dtype=DTYPE, device=DEVICE).reshape(-1, 2)
        theta = torch.as_tensor(headings, dtype=DTYPE, device=DEVICE)
        return OpenSkySnapshot(
            time_position=sanitized["api_time"],
            euclidean=x,
            heading=theta,
            track_keys=tuple(row["track_key"] for row in observations),
            categories=np.asarray(
                [
                    -1 if row["category"] is None else row["category"]
                    for row in observations
                ],
                dtype=np.int64,
            ),
            network_latency_ms=metadata["network_latency_ms"],
        )


def nested_keep_mask(n: int, dropout_rate: float, block_seed: int) -> np.ndarray:
    """Create reproducible nested observation-thinning masks for a block."""

    integer(n, "observation count")
    integer(block_seed, "block_seed")
    finite_real(dropout_rate, "dropout_rate")
    if not dropout_rate < 1.0:
        raise ValueError("dropout_rate must lie in [0, 1)")
    uniforms = np.random.default_rng(block_seed).random(n)
    return uniforms >= dropout_rate


def _log_i0(x: torch.Tensor) -> torch.Tensor:
    """Stable log(I_0(x)) for nonnegative x."""

    return torch.log(torch.special.i0e(x)) + torch.abs(x)


def _a1(x: torch.Tensor) -> torch.Tensor:
    """Stable I_1(x) / I_0(x), including the value zero at x=0."""

    ratio = torch.special.i1e(x) / torch.special.i0e(x)
    return torch.where(x == 0, torch.zeros_like(ratio), ratio)


def _model_data(
    x: torch.Tensor, theta: torch.Tensor, minimum: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(x, torch.Tensor) or not isinstance(theta, torch.Tensor):
        raise ValueError("model inputs must be tensors")
    if x.ndim != 2 or x.shape[1] != 2 or theta.shape != (x.shape[0],):
        raise ValueError("expected x=[N,2] and theta=[N]")
    if x.shape[0] < minimum or x.is_complex() or theta.is_complex():
        raise ValueError("insufficient or non-real model observations")
    x = x.detach().to(device=DEVICE, dtype=DTYPE)
    theta = theta.detach().to(device=DEVICE, dtype=DTYPE)
    if not torch.isfinite(x).all() or not torch.isfinite(theta).all():
        raise ValueError("model observations must be finite")
    return x, theta


def _base_moments(x: torch.Tensor, mean, variance) -> tuple[torch.Tensor, torch.Tensor]:
    if (mean is None) != (variance is None):
        raise ValueError("provide both fixed base mean and variance")
    if mean is None:
        return x.mean(0), x.var(0, unbiased=True).clamp_min(torch.finfo(DTYPE).eps)
    m = torch.as_tensor(mean, dtype=DTYPE, device=DEVICE)
    v = torch.as_tensor(variance, dtype=DTYPE, device=DEVICE)
    if (
        m.shape != (2,)
        or v.shape != (2,)
        or not torch.isfinite(m).all()
        or not torch.isfinite(v).all()
        or not (v > 0).all()
    ):
        raise ValueError(
            "fixed Normal-Gamma base needs two finite means and positive variances"
        )
    return m.detach().clone(), v.detach().clone()


class StreamingCircularDPMM:
    """Truncated streaming VB mixture on R^2 x S^1.

    Euclidean coordinates use Normal-Gamma variational factors.  Heading uses a
    von Mises likelihood with a von Mises variational factor for its component
    direction.  ``forgetting=1`` accumulates all sufficient statistics;
    ``forgetting<1`` defines a power-prior tracking target and must not be sold
    as ordinary stationary posterior consistency.
    """

    def __init__(
        self,
        max_clusters: int = 12,
        alpha: float = 1.5,
        heading_kappa: float = 6.0,
        forgetting: float = 1.0,
        initialization_mass: float = 0.25,
        base_mean: Iterable[float] | None = None,
        base_variance: Iterable[float] | None = None,
    ) -> None:
        integer(max_clusters, "max_clusters", 2)
        finite_real(alpha, "alpha", positive=True)
        finite_real(heading_kappa, "heading_kappa", positive=True)
        finite_real(initialization_mass, "initialization_mass", positive=True)
        finite_real(forgetting, "forgetting", positive=True)
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must lie in (0, 1]")
        self.K = max_clusters
        self.alpha = float(alpha)
        self.heading_kappa = torch.tensor(heading_kappa, dtype=DTYPE, device=DEVICE)
        self.forgetting = float(forgetting)
        self.initialization_mass = float(initialization_mass)
        self.base_mean, self.base_variance = base_mean, base_variance
        self.initialized = False

    def initialize(self, x: torch.Tensor, theta: torch.Tensor) -> None:
        x, theta = _model_data(x, theta)
        if x.shape[0] < self.K:
            raise ValueError(
                "the calibration batch must contain at least K observations"
            )

        self.m0, empirical_var = _base_moments(x, self.base_mean, self.base_variance)
        self.kappa0 = torch.tensor(0.05, dtype=DTYPE, device=DEVICE)
        self.a0 = torch.full((2,), 2.5, dtype=DTYPE, device=DEVICE)
        self.b0 = empirical_var * (self.a0 - 1.0)

        order = torch.argsort(x[:, 0])
        positions = (
            torch.linspace(0, x.shape[0] - 1, self.K, device=DEVICE).round().long()
        )
        seeds = order[positions]
        centers = x[seeds]
        seed_theta = theta[seeds]
        mass = self.initialization_mass
        self.counts = torch.full((self.K,), mass, dtype=DTYPE, device=DEVICE)
        self.sums = mass * centers
        self.sumsq = mass * centers.square()
        self.direction = mass * torch.stack(
            [torch.cos(seed_theta), torch.sin(seed_theta)], dim=1
        )
        self._refresh_variational_parameters()
        self._bootstrap_only = True
        self.initialized = True

    def _refresh_variational_parameters(self) -> None:
        self.stick_a = 1.0 + self.counts[:-1]
        tail_counts = torch.flip(
            torch.cumsum(torch.flip(self.counts[1:], dims=[0]), dim=0), dims=[0]
        )
        self.stick_b = self.alpha + tail_counts

        self.post_kappa = self.kappa0 + self.counts[:, None]
        self.post_mean = (self.kappa0 * self.m0[None, :] + self.sums) / self.post_kappa
        self.post_a = self.a0[None, :] + 0.5 * self.counts[:, None]
        quadratic = (
            self.sumsq
            + self.kappa0 * self.m0[None, :].square()
            - self.post_kappa * self.post_mean.square()
        )
        self.post_b = self.b0[None, :] + 0.5 * quadratic
        self.post_b = self.post_b.clamp_min(torch.finfo(DTYPE).tiny)

        self.direction_eta = self.heading_kappa * self.direction
        self.direction_r = torch.linalg.vector_norm(self.direction_eta, dim=1)
        self.direction_mean = torch.atan2(
            self.direction_eta[:, 1], self.direction_eta[:, 0]
        )

    def expected_log_weights(self) -> torch.Tensor:
        elog_v = torch.digamma(self.stick_a) - torch.digamma(
            self.stick_a + self.stick_b
        )
        elog_1mv = torch.digamma(self.stick_b) - torch.digamma(
            self.stick_a + self.stick_b
        )
        prefix = torch.cat(
            [
                torch.zeros(1, dtype=DTYPE, device=DEVICE),
                torch.cumsum(elog_1mv, dim=0),
            ]
        )
        result = prefix.clone()
        result[:-1] += elog_v
        return result

    def expected_weights(self) -> torch.Tensor:
        if self.initialized:
            mean_v = self.stick_a / (self.stick_a + self.stick_b)
            mean_1mv = self.stick_b / (self.stick_a + self.stick_b)
        else:
            # Prior expectation under V_k ~ Beta(1, alpha), exposed so the
            # truncation error can be audited before a calibration batch exists.
            mean_v = torch.full(
                (self.K - 1,), 1.0 / (1.0 + self.alpha), dtype=DTYPE, device=DEVICE
            )
            mean_1mv = torch.full(
                (self.K - 1,),
                self.alpha / (1.0 + self.alpha),
                dtype=DTYPE,
                device=DEVICE,
            )
        weights = torch.empty(self.K, dtype=DTYPE, device=DEVICE)
        residual = torch.tensor(1.0, dtype=DTYPE, device=DEVICE)
        for k in range(self.K - 1):
            weights[k] = residual * mean_v[k]
            residual = residual * mean_1mv[k]
        weights[-1] = residual
        return weights

    def _expected_log_likelihood(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        expected_precision = self.post_a / self.post_b
        expected_log_precision = torch.digamma(self.post_a) - torch.log(self.post_b)
        squared = (x[:, None, :] - self.post_mean[None, :, :]).square()
        euclidean = 0.5 * (
            expected_log_precision[None, :, :]
            - LOG_2PI
            - 1.0 / self.post_kappa[None, :, :]
            - expected_precision[None, :, :] * squared
        ).sum(dim=2)

        circular_concentration = _a1(self.direction_r)
        circular = (
            self.heading_kappa
            * circular_concentration[None, :]
            * torch.cos(theta[:, None] - self.direction_mean[None, :])
            - LOG_2PI
            - _log_i0(self.heading_kappa)
        )
        return euclidean + circular

    def responsibilities(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            raise RuntimeError("call initialize on training data first")
        x, theta = _model_data(x, theta, minimum=0)
        logits = self._expected_log_likelihood(x, theta)
        logits = logits + self.expected_log_weights()[None, :]
        return torch.softmax(logits, dim=1)

    def log_predictive(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Approximate held-forward posterior predictive log density.

        Evaluate this before the corresponding streaming update.  The
        Euclidean term is the Normal-Gamma Student-t predictive and the
        directional term integrates the von Mises likelihood against its
        posterior on the circle.  Components are averaged with the residual-
        stick posterior mean weights.
        """

        if not self.initialized:
            raise RuntimeError("call initialize on a calibration batch first")
        x, theta = _model_data(x, theta, minimum=0)
        degrees = 2.0 * self.post_a
        scale = torch.sqrt(
            self.post_b * (self.post_kappa + 1.0) / (self.post_a * self.post_kappa)
        )
        euclidean = (
            torch.distributions.StudentT(
                degrees[None, :, :],
                self.post_mean[None, :, :],
                scale[None, :, :],
            )
            .log_prob(x[:, None, :])
            .sum(dim=2)
        )

        unit = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        updated_eta = (
            self.direction_eta[None, :, :] + self.heading_kappa * unit[:, None, :]
        )
        circular = (
            _log_i0(torch.linalg.vector_norm(updated_eta, dim=2))
            - _log_i0(self.direction_r)[None, :]
            - LOG_2PI
            - _log_i0(self.heading_kappa)
        )
        component_log_density = euclidean + circular
        log_weights = torch.log(
            self.expected_weights().clamp_min(torch.finfo(DTYPE).tiny)
        )
        return torch.logsumexp(
            log_weights[None, :] + component_log_density,
            dim=1,
        )

    def partial_fit(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x, theta = _model_data(x, theta)
        if not self.initialized:
            self.initialize(x, theta)
        responsibilities = self.responsibilities(x, theta)
        batch_counts = responsibilities.sum(0)
        batch_sums = responsibilities.T @ x
        batch_sumsq = responsibilities.T @ x.square()
        unit = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        batch_direction = responsibilities.T @ unit

        if self._bootstrap_only:
            # The seed locations break label symmetry for the first E-step but
            # are not retained as pseudo-observations in the posterior target.
            self.counts = batch_counts
            self.sums = batch_sums
            self.sumsq = batch_sumsq
            self.direction = batch_direction
            self._bootstrap_only = False
        else:
            decay = self.forgetting
            self.counts = decay * self.counts + batch_counts
            self.sums = decay * self.sums + batch_sums
            self.sumsq = decay * self.sumsq + batch_sumsq
            self.direction = decay * self.direction + batch_direction
        self._refresh_variational_parameters()
        return responsibilities

    def active_clusters(self, minimum_weight: float = 0.01) -> int:
        return int((self.expected_weights() >= minimum_weight).sum().item())


@dataclass
class _ClusterStats:
    n: float
    sums: torch.Tensor
    sumsq: torch.Tensor
    direction: torch.Tensor

    @classmethod
    def empty(cls) -> "_ClusterStats":
        return cls(
            0.0,
            torch.zeros(2, dtype=DTYPE, device=DEVICE),
            torch.zeros(2, dtype=DTYPE, device=DEVICE),
            torch.zeros(2, dtype=DTYPE, device=DEVICE),
        )

    def clone(self) -> "_ClusterStats":
        return _ClusterStats(
            self.n, self.sums.clone(), self.sumsq.clone(), self.direction.clone()
        )

    def add(self, x: torch.Tensor, theta: torch.Tensor) -> None:
        self.n += 1.0
        self.sums += x
        self.sumsq += x.square()
        self.direction += torch.stack([torch.cos(theta), torch.sin(theta)])

    def remove(self, x: torch.Tensor, theta: torch.Tensor) -> None:
        self.n -= 1.0
        self.sums -= x
        self.sumsq -= x.square()
        self.direction -= torch.stack([torch.cos(theta), torch.sin(theta)])


@dataclass
class _Particle:
    clusters: dict[int, _ClusterStats] = field(default_factory=dict)
    assignments: list[int] = field(default_factory=list)
    x_history: list[torch.Tensor] = field(default_factory=list)
    theta_history: list[torch.Tensor] = field(default_factory=list)
    next_id: int = 0
    log_weight: float = 0.0

    def clone(self) -> "_Particle":
        return _Particle(
            clusters={key: value.clone() for key, value in self.clusters.items()},
            assignments=list(self.assignments),
            x_history=[value.clone() for value in self.x_history],
            theta_history=[value.clone() for value in self.theta_history],
            next_id=self.next_id,
            log_weight=self.log_weight,
        )


class CollapsedDPSMC:
    """Collapsed DP particle filter with systematic resampling and Gibbs moves."""

    def __init__(
        self,
        calibration_x: torch.Tensor,
        particles: int = 64,
        alpha: float = 1.5,
        heading_kappa: float = 6.0,
        resample_fraction: float = 0.5,
        rejuvenation_window: int = 24,
        seed: int = 2026,
        base_mean: Iterable[float] | None = None,
        base_variance: Iterable[float] | None = None,
    ) -> None:
        integer(particles, "particles", 2)
        integer(rejuvenation_window, "rejuvenation_window", 0)
        integer(seed, "seed")
        finite_real(alpha, "alpha", positive=True)
        finite_real(heading_kappa, "heading_kappa", positive=True)
        finite_real(resample_fraction, "resample_fraction", positive=True)
        if resample_fraction > 1:
            raise ValueError("resample_fraction must lie in (0, 1]")
        if not isinstance(calibration_x, torch.Tensor) or calibration_x.ndim != 2:
            raise ValueError("expected calibration_x=[N,2]")
        calibration_x, _ = _model_data(
            calibration_x,
            torch.zeros(calibration_x.shape[0], dtype=DTYPE, device=DEVICE),
            2,
        )
        self.P = particles
        self.alpha = float(alpha)
        self.heading_kappa = torch.tensor(heading_kappa, dtype=DTYPE, device=DEVICE)
        self.resample_threshold = resample_fraction * particles
        self.window = rejuvenation_window
        self.rng = torch.Generator(device=DEVICE).manual_seed(seed)
        self.m0, empirical_var = _base_moments(calibration_x, base_mean, base_variance)
        self.kappa0 = torch.tensor(0.05, dtype=DTYPE, device=DEVICE)
        self.a0 = torch.full((2,), 2.5, dtype=DTYPE, device=DEVICE)
        self.b0 = empirical_var * (self.a0 - 1.0)
        initial_log_weight = -math.log(particles)
        self.particles = [
            _Particle(log_weight=initial_log_weight) for _ in range(particles)
        ]
        self.ess_history: list[float] = []
        self.log_score_history: list[float] = []

    def _posterior(self, stats: _ClusterStats) -> tuple[torch.Tensor, ...]:
        kappa = self.kappa0 + stats.n
        mean = (self.kappa0 * self.m0 + stats.sums) / kappa
        a = self.a0 + 0.5 * stats.n
        b = self.b0 + 0.5 * (
            stats.sumsq + self.kappa0 * self.m0.square() - kappa * mean.square()
        )
        return kappa, mean, a, b.clamp_min(torch.finfo(DTYPE).tiny)

    def _log_predictive(
        self, x: torch.Tensor, theta: torch.Tensor, stats: _ClusterStats
    ) -> torch.Tensor:
        kappa, mean, a, b = self._posterior(stats)
        degrees = 2.0 * a
        scale = torch.sqrt(b * (kappa + 1.0) / (a * kappa))
        euclidean = torch.distributions.StudentT(degrees, mean, scale).log_prob(x).sum()

        eta = self.heading_kappa * stats.direction
        unit = torch.stack([torch.cos(theta), torch.sin(theta)])
        circular = (
            _log_i0(torch.linalg.vector_norm(eta + self.heading_kappa * unit))
            - _log_i0(torch.linalg.vector_norm(eta))
            - LOG_2PI
            - _log_i0(self.heading_kappa)
        )
        return euclidean + circular

    def _draw_assignment(
        self, particle: _Particle, x: torch.Tensor, theta: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        cluster_ids = sorted(particle.clusters)
        log_options = []
        for cluster_id in cluster_ids:
            stats = particle.clusters[cluster_id]
            log_options.append(
                math.log(stats.n) + self._log_predictive(x, theta, stats)
            )
        log_options.append(
            math.log(self.alpha) + self._log_predictive(x, theta, _ClusterStats.empty())
        )
        logits = torch.stack(log_options)
        probabilities = torch.softmax(logits, dim=0)
        selected = int(torch.multinomial(probabilities, 1, generator=self.rng).item())
        if selected == len(cluster_ids):
            cluster_id = particle.next_id
            particle.next_id += 1
            particle.clusters[cluster_id] = _ClusterStats.empty()
        else:
            cluster_id = cluster_ids[selected]
        return cluster_id, torch.logsumexp(logits, dim=0)

    def _normalize(self) -> tuple[torch.Tensor, float]:
        log_weights = torch.tensor(
            [particle.log_weight for particle in self.particles],
            dtype=DTYPE,
            device=DEVICE,
        )
        log_weights -= torch.logsumexp(log_weights, dim=0)
        weights = torch.exp(log_weights)
        for particle, value in zip(self.particles, log_weights.tolist()):
            particle.log_weight = float(value)
        ess = float(1.0 / weights.square().sum().item())
        return weights, ess

    def _systematic_resample(self, weights: torch.Tensor) -> None:
        start = torch.rand((), generator=self.rng, device=DEVICE) / self.P
        positions = start + torch.arange(self.P, dtype=DTYPE, device=DEVICE) / self.P
        cumulative = torch.cumsum(weights, dim=0)
        cumulative[-1] = 1.0
        indices = torch.searchsorted(cumulative, positions).tolist()
        reset = -math.log(self.P)
        self.particles = [self.particles[index].clone() for index in indices]
        for particle in self.particles:
            particle.log_weight = reset

    def _rejuvenate(self, particle: _Particle) -> None:
        start = max(0, len(particle.assignments) - self.window)
        for index in range(start, len(particle.assignments)):
            x = particle.x_history[index]
            theta = particle.theta_history[index]
            old_id = particle.assignments[index]
            old_stats = particle.clusters[old_id]
            old_stats.remove(x, theta)
            if old_stats.n <= 0.0:
                del particle.clusters[old_id]
            new_id, _ = self._draw_assignment(particle, x, theta)
            particle.clusters[new_id].add(x, theta)
            particle.assignments[index] = new_id

    def partial_fit(self, x: torch.Tensor, theta: torch.Tensor) -> float:
        x, theta = _model_data(x, theta)
        for observation, angle in zip(x, theta):
            previous_n = len(self.particles[0].assignments)
            denominator = math.log(previous_n + self.alpha)
            weights_before, _ = self._normalize()
            increments: list[float] = []
            for particle in self.particles:
                cluster_id, log_evidence = self._draw_assignment(
                    particle, observation, angle
                )
                particle.clusters[cluster_id].add(observation, angle)
                particle.assignments.append(cluster_id)
                particle.x_history.append(observation.clone())
                particle.theta_history.append(angle.clone())
                increment = float(log_evidence.item()) - denominator
                increments.append(increment)
                particle.log_weight += increment
            log_predictive = torch.logsumexp(
                torch.log(weights_before.clamp_min(torch.finfo(DTYPE).tiny))
                + torch.as_tensor(increments, dtype=DTYPE, device=DEVICE),
                dim=0,
            )
            self.log_score_history.append(float(log_predictive.item()))
            weights, ess = self._normalize()
            # Preserve the degeneracy that triggered resampling, rather than
            # replacing it by the mechanically uniform post-resampling weights.
            self.ess_history.append(ess)
            if ess < self.resample_threshold:
                self._systematic_resample(weights)
                for particle in self.particles:
                    self._rejuvenate(particle)
                self._normalize()
        return self.ess_history[-1] if self.ess_history else float(self.P)

    def log_predictive(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Marginal densities under one frozen pre-snapshot posterior.

        This query does not normalize in place, draw allocations, or update
        histories. Every aircraft in a snapshot sees the same past information.
        """

        x, theta = _model_data(x, theta, minimum=0)
        log_weights = torch.as_tensor(
            [p.log_weight for p in self.particles], dtype=DTYPE, device=DEVICE
        )
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        scores: list[torch.Tensor] = []
        for particle in self.particles:
            options = []
            for stats in particle.clusters.values():
                options.append(
                    math.log(stats.n) + self._log_predictive_batch(x, theta, stats)
                )
            options.append(
                math.log(self.alpha)
                + self._log_predictive_batch(x, theta, _ClusterStats.empty())
            )
            scores.append(
                torch.logsumexp(torch.stack(options), dim=0)
                - math.log(len(particle.assignments) + self.alpha)
            )
        return torch.logsumexp(log_weights[:, None] + torch.stack(scores), dim=0)

    def _log_predictive_batch(
        self, x: torch.Tensor, theta: torch.Tensor, stats: _ClusterStats
    ) -> torch.Tensor:
        kappa, mean, a, b = self._posterior(stats)
        scale = torch.sqrt(b * (kappa + 1.0) / (a * kappa))
        euclidean = (
            torch.distributions.StudentT(2.0 * a, mean, scale).log_prob(x).sum(dim=1)
        )
        eta = self.heading_kappa * stats.direction
        unit = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        circular = (
            _log_i0(
                torch.linalg.vector_norm(
                    eta[None, :] + self.heading_kappa * unit, dim=1
                )
            )
            - _log_i0(torch.linalg.vector_norm(eta))
            - LOG_2PI
            - _log_i0(self.heading_kappa)
        )
        return euclidean + circular

    def map_partition(self) -> np.ndarray:
        weights, _ = self._normalize()
        selected = int(torch.argmax(weights).item())
        return np.asarray(self.particles[selected].assignments, dtype=np.int64)


def synchronized_latency_ms(function, *args, **kwargs):
    """Measure compute latency separately from API/network latency."""

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return result, 1000.0 * (time.perf_counter() - started)


def _split_chains(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2 or chains.shape[0] < 4 or chains.shape[1] < 20:
        raise ValueError("publication diagnostics require at least 4 chains x 20 draws")
    if not np.isfinite(chains).all():
        raise ValueError("diagnostic chains must contain finite draws")
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)


def _basic_rhat(chains: np.ndarray) -> float:
    m, n = chains.shape
    within = np.mean(np.var(chains, axis=1, ddof=1))
    between = n * np.var(np.mean(chains, axis=1), ddof=1)
    if within == 0:
        if between > 0:
            return float("inf")
        raise ValueError("constant diagnostic draws have no estimable variance")
    variance = ((n - 1.0) / n) * within + between / n

    if variance <= 0:
        raise ValueError("constant diagnostic draws have no estimable variance")
    return float(np.sqrt(variance / within))


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    ranks = rankdata(flat, method="average")
    probabilities = (ranks - 3.0 / 8.0) / (len(flat) + 1.0 / 4.0)
    return norm.ppf(probabilities).reshape(values.shape)


def rank_normalized_split_rhat(chains: np.ndarray) -> dict[str, float]:
    """Return bulk, folded, and recommended max rank-normalized split R-hat."""

    split = _split_chains(chains)
    bulk = _basic_rhat(_rank_normalize(split))
    folded = np.abs(split - np.median(split))
    folded_rhat = _basic_rhat(_rank_normalize(folded))
    return {"bulk": bulk, "folded": folded_rhat, "max": max(bulk, folded_rhat)}


def bulk_ess(chains: np.ndarray) -> float:
    """Rank-normalized multi-chain bulk ESS with Geyer's positive sequence."""

    z = _rank_normalize(_split_chains(chains))
    m, n = z.shape
    chain_var = np.var(z, axis=1, ddof=1)
    within = float(np.mean(chain_var))
    between = n * float(np.var(np.mean(z, axis=1), ddof=1))
    variance = ((n - 1.0) / n) * within + between / n

    autocorrelation = [1.0]
    if variance <= 0:
        raise ValueError("constant diagnostic draws have no estimable variance")
    centered = z - z.mean(axis=1, keepdims=True)
    for lag in range(1, n):
        autocovariance = np.mean(
            np.sum(centered[:, : n - lag] * centered[:, lag:], axis=1) / n
        )
        autocorrelation.append(1.0 - (within - autocovariance) / variance)

    pair_sums: list[float] = []
    for lag in range(0, n - 1, 2):
        pair = autocorrelation[lag] + autocorrelation[lag + 1]
        if pair < 0.0:
            break
        if pair_sums:
            pair = min(pair, pair_sums[-1])
        pair_sums.append(pair)
    tau = max(-1.0 + 2.0 * sum(pair_sums), 1.0)
    return float(min(m * n, m * n / tau))


def self_test() -> dict[str, float]:
    generator = torch.Generator(device=DEVICE).manual_seed(2026)
    n = 120
    centers = torch.tensor(
        [[4.6, -0.8], [5.2, 0.0], [5.7, 0.9]], dtype=DTYPE, device=DEVICE
    )
    directions = torch.tensor([0.2, 2.3, 5.0], dtype=DTYPE, device=DEVICE)
    labels = torch.arange(n, device=DEVICE) % 3
    x = centers[labels] + 0.15 * torch.randn(
        n, 2, dtype=DTYPE, device=DEVICE, generator=generator
    )
    theta = torch.remainder(
        directions[labels]
        + 0.20 * torch.randn(n, dtype=DTYPE, device=DEVICE, generator=generator),
        2.0 * math.pi,
    )

    vi = StreamingCircularDPMM(max_clusters=6)
    responsibilities, vi_ms = synchronized_latency_ms(vi.partial_fit, x, theta)
    if not torch.allclose(
        responsibilities.sum(1), torch.ones(n, dtype=DTYPE, device=DEVICE)
    ):
        raise AssertionError("responsibilities do not normalize")
    if not torch.allclose(
        vi.expected_weights().sum(), torch.tensor(1.0, dtype=DTYPE, device=DEVICE)
    ):
        raise AssertionError("truncated weights do not include the residual stick")

    smc = CollapsedDPSMC(x[:30], particles=24, rejuvenation_window=8)
    smc_ess, smc_ms = synchronized_latency_ms(smc.partial_fit, x[:45], theta[:45])
    if not (1.0 <= smc_ess <= smc.P):
        raise AssertionError("invalid particle ESS")

    rng = np.random.default_rng(2026)
    chains = rng.normal(size=(4, 400))
    rhat = rank_normalized_split_rhat(chains)
    ess = bulk_ess(chains)
    if not np.isfinite([*rhat.values(), ess]).all():
        raise AssertionError("non-finite diagnostics")
    return {
        "vi_active_clusters": float(vi.active_clusters()),
        "vi_latency_ms": vi_ms,
        "smc_particle_ess": smc_ess,
        "smc_latency_ms": smc_ms,
        "rhat_max": rhat["max"],
        "bulk_ess": ess,
    }


def run_live_example(dropout_levels: Iterable[float]) -> list[dict[str, float]]:
    client = OpenSkyClient(
        client_id=os.getenv("OPENSKY_CLIENT_ID"),
        client_secret=os.getenv("OPENSKY_CLIENT_SECRET"),
    )
    snapshot = client.fetch()
    x, theta = snapshot.model_view()
    if x.shape[0] < 12:
        raise OpenSkyError("too few complete states in the selected bounding box")

    records = []
    for rate in dropout_levels:
        keep = nested_keep_mask(x.shape[0], rate, block_seed=snapshot.time_position)
        index = torch.as_tensor(np.flatnonzero(keep), dtype=torch.long, device=DEVICE)
        x_sub, theta_sub = x[index], theta[index]
        model = StreamingCircularDPMM(max_clusters=min(12, x_sub.shape[0]))
        _, compute_ms = synchronized_latency_ms(model.partial_fit, x_sub, theta_sub)
        records.append(
            {
                "dropout_rate": float(rate),
                "retained_observations": float(x_sub.shape[0]),
                "active_clusters": float(model.active_clusters()),
                "network_latency_ms_shared": snapshot.network_latency_ms,
                "vi_compute_latency_ms": compute_ms,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true", help="run one descriptive live pull"
    )
    args = parser.parse_args()
    result = run_live_example([0.0, 0.15, 0.30]) if args.live else self_test()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
