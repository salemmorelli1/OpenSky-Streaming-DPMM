import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import i0e

from opensky_streaming_dpmm.empirical import _fit_smc, _fit_vi
from opensky_streaming_dpmm.engine import (
    DEVICE,
    DTYPE,
    CollapsedDPSMC,
    StreamingCircularDPMM,
    _ClusterStats,
    bulk_ess,
    rank_normalized_split_rhat,
)
from test_empirical import snapshots


CONFIG = dict(
    selected_K=4, alpha=1.5, heading_kappa=6.0, particles=4, rejuvenation_window=4
)


def test_smc_predictive_query_is_frozen_and_order_invariant():
    first, later = snapshots((10, 7))
    model = CollapsedDPSMC(first.x, particles=4, rejuvenation_window=4, seed=99)
    model.partial_fit(first.x, first.theta)
    rng = model.rng.get_state().clone()
    before = [
        (
            p.log_weight,
            list(p.assignments),
            {k: s.clone() for k, s in p.clusters.items()},
        )
        for p in model.particles
    ]
    scores = model.log_predictive(later.x, later.theta)
    order = torch.tensor([6, 0, 2, 1, 3, 5, 4], device=DEVICE)
    torch.testing.assert_close(
        scores[order], model.log_predictive(later.x[order], later.theta[order])
    )
    torch.testing.assert_close(model.rng.get_state(), rng)
    for particle, (weight, assignments, stats) in zip(model.particles, before):
        assert particle.log_weight == weight and particle.assignments == assignments
        assert set(particle.clusters) == set(stats)
        for k, s in stats.items():
            assert particle.clusters[k].n == s.n
            torch.testing.assert_close(particle.clusters[k].sums, s.sums)
            torch.testing.assert_close(particle.clusters[k].direction, s.direction)
    assert len(model.log_score_history) == first.x.shape[0]


def test_smc_predictive_matches_weighted_collapsed_crp_mixture():
    first, later = snapshots((8, 3))
    model = CollapsedDPSMC(first.x, particles=4, seed=81)
    model.partial_fit(first.x, first.theta)
    # Explicit scalar calculation independently checks the particle and CRP
    # normalization of the vectorized read-only prediction path.
    log_weights = torch.tensor(
        [p.log_weight for p in model.particles], dtype=DTYPE, device=DEVICE
    )
    log_weights -= torch.logsumexp(log_weights, dim=0)
    expected = []
    for x, theta in zip(later.x, later.theta):
        particle_scores = []
        for particle in model.particles:
            options = [
                math.log(s.n) + model._log_predictive(x, theta, s)
                for s in particle.clusters.values()
            ]
            options.append(
                math.log(model.alpha)
                + model._log_predictive(x, theta, _ClusterStats.empty())
            )
            particle_scores.append(
                torch.logsumexp(torch.stack(options), 0)
                - math.log(len(particle.assignments) + model.alpha)
            )
        expected.append(torch.logsumexp(log_weights + torch.stack(particle_scores), 0))
    torch.testing.assert_close(
        model.log_predictive(later.x, later.theta), torch.stack(expected)
    )


def test_runner_scores_smc_before_any_current_snapshot_updates():
    data = snapshots((10, 7, 5))
    model = CollapsedDPSMC(
        data[0].x,
        particles=4,
        alpha=1.5,
        heading_kappa=6,
        rejuvenation_window=4,
        seed=91,
    )
    expected = []
    for i, snapshot in enumerate(data):
        if i:
            expected.append(
                model.log_predictive(snapshot.x, snapshot.theta).cpu().numpy()
            )
        model.partial_fit(snapshot.x, snapshot.theta)
    result = _fit_smc(data, CONFIG, 91)
    np.testing.assert_allclose(result["scores"], np.concatenate(expected))
    # This would differ if the scorer used within-snapshot sequential increments.
    assert not np.allclose(result["scores"], np.asarray(model.log_score_history[10:]))


def test_vi_final_partition_and_count_use_same_allocations():
    data = snapshots((10, 7))
    result = _fit_vi(data, CONFIG)
    model = StreamingCircularDPMM(max_clusters=4, alpha=1.5, heading_kappa=6)
    for snapshot in data:
        model.partial_fit(snapshot.x, snapshot.theta)
    labels = np.concatenate(
        [model.responsibilities(s.x, s.theta).argmax(1).cpu().numpy() for s in data]
    )
    np.testing.assert_array_equal(result["labels"], labels)
    assert result["occupied_clusters"] == np.unique(labels).size


@pytest.mark.parametrize("method", ["vi", "smc"])
def test_categories_cannot_change_inference_or_predictive_score(method):
    from opensky_streaming_dpmm.empirical import SnapshotData

    data = snapshots((10, 7))
    changed = tuple(
        SnapshotData(s.x, s.theta, np.full(len(s.category), 20)) for s in data
    )
    fit = (
        (lambda d: _fit_smc(d, CONFIG, 77))
        if method == "smc"
        else (lambda d: _fit_vi(d, CONFIG))
    )
    first, second = fit(data), fit(changed)
    np.testing.assert_array_equal(first["labels"], second["labels"])
    np.testing.assert_array_equal(first["scores"], second["scores"])


def test_circular_predictive_density_normalizes_under_arc_length_measure():
    kappa = 6.0
    eta = np.asarray([12.0, -7.0])

    def log_i0(x):
        return np.log(i0e(x)) + abs(x)

    def density(theta):
        u = np.asarray([math.cos(theta), math.sin(theta)])
        return math.exp(
            log_i0(np.linalg.norm(eta + kappa * u))
            - log_i0(np.linalg.norm(eta))
            - log_i0(kappa)
            - math.log(2 * math.pi)
        )

    area, _ = quad(density, 0, 2 * math.pi, epsabs=1e-10)
    assert area == pytest.approx(1, abs=1e-9)
    first, _ = snapshots((10, 7))
    model = CollapsedDPSMC(first.x, particles=4, heading_kappa=kappa)
    stats = _ClusterStats.empty()
    stats.direction = torch.tensor(eta / kappa, dtype=DTYPE, device=DEVICE)
    x = first.x[0]
    k, mean, a, b = model._posterior(stats)
    euclidean = (
        torch.distributions.StudentT(2 * a, mean, torch.sqrt(b * (k + 1) / (a * k)))
        .log_prob(x)
        .sum()
    )
    theta = torch.tensor(0.7, dtype=DTYPE, device=DEVICE)
    assert math.exp(
        float(model._log_predictive(x, theta, stats) - euclidean)
    ) == pytest.approx(density(0.7))


@pytest.mark.parametrize("method", ["vi", "smc"])
def test_heading_periodicity_in_both_predictives(method):
    first, later = snapshots((10, 7))
    model = (
        StreamingCircularDPMM(max_clusters=4)
        if method == "vi"
        else CollapsedDPSMC(first.x, particles=4)
    )
    model.partial_fit(first.x, first.theta)
    torch.testing.assert_close(
        model.log_predictive(later.x, later.theta),
        model.log_predictive(later.x, later.theta + 2 * math.pi),
    )


def test_vi_accumulates_each_observation_once_and_honors_forgetting():
    first, later = snapshots((10, 7))
    for decay in (1.0, 0.5):
        model = StreamingCircularDPMM(max_clusters=4, forgetting=decay)
        model.partial_fit(first.x, first.theta)
        assert model.counts.sum().item() == pytest.approx(10)
        model.partial_fit(later.x, later.theta)
        assert model.counts.sum().item() == pytest.approx(decay * 10 + 7)
        assert model.stick_a.shape == model.stick_b.shape == (3,)
        assert model.expected_weights().sum().item() == pytest.approx(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(alpha=float("nan")),
        dict(alpha=-1),
        dict(heading_kappa=float("inf")),
        dict(max_clusters=2.5),
        dict(max_clusters=True),
        dict(initialization_mass=0),
    ],
)
def test_vi_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        StreamingCircularDPMM(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(alpha=0),
        dict(heading_kappa=-1),
        dict(particles=2.5),
        dict(rejuvenation_window=-1),
        dict(resample_fraction=2),
        dict(resample_fraction=float("nan")),
    ],
)
def test_smc_rejects_invalid_hyperparameters(kwargs):
    first, _ = snapshots((10, 7))
    with pytest.raises(ValueError):
        CollapsedDPSMC(first.x, **kwargs)


@pytest.mark.parametrize("method", ["vi", "smc"])
def test_mismatched_or_nonfinite_batches_cannot_partially_mutate_model(method):
    first, later = snapshots((10, 7))
    model = (
        StreamingCircularDPMM(max_clusters=4)
        if method == "vi"
        else CollapsedDPSMC(first.x, particles=4)
    )
    model.partial_fit(first.x, first.theta)
    before = (
        model.counts.sum().item()
        if method == "vi"
        else len(model.particles[0].assignments)
    )
    with pytest.raises(ValueError, match="theta"):
        model.partial_fit(later.x, later.theta[:2])
    bad = later.x.clone()
    bad[-1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.partial_fit(bad, later.theta)
    after = (
        model.counts.sum().item()
        if method == "vi"
        else len(model.particles[0].assignments)
    )
    assert before == after


def test_particle_health_records_pre_resampling_degeneracy():
    first, _ = snapshots((10, 7))
    model = CollapsedDPSMC(
        first.x, particles=4, resample_fraction=1, rejuvenation_window=0
    )
    model.particles[0].log_weight = 0
    for particle in model.particles[1:]:
        particle.log_weight = -20
    with patch.object(
        model, "_systematic_resample", wraps=model._systematic_resample
    ) as resample:
        model.partial_fit(first.x[:1], first.theta[:1])
    assert resample.call_count == 1
    assert model.ess_history[-1] < 1.01
    weights, ess = model._normalize()
    assert ess == pytest.approx(4)
    assert weights.tolist() == pytest.approx([0.25] * 4)


@pytest.mark.parametrize("diagnostic", [rank_normalized_split_rhat, bulk_ess])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0])
def test_bad_diagnostic_traces_cannot_look_converged(diagnostic, value):
    with pytest.raises(ValueError):
        diagnostic(np.full((4, 100), value))


def test_autocorrelated_chains_have_lower_bulk_ess():
    rng = np.random.default_rng(77)
    independent = rng.normal(size=(4, 1000))
    correlated = np.zeros_like(independent)
    for t in range(1, correlated.shape[1]):
        correlated[:, t] = 0.9 * correlated[:, t - 1] + independent[:, t]
    assert bulk_ess(correlated) < bulk_ess(independent) / 5


@pytest.mark.parametrize("method", ["vi", "smc"])
def test_fixed_calibration_base_is_identical_across_different_initial_samples(method):
    first, later = snapshots((10, 7))
    means, variances = [4.5, -0.3], [0.4, 1.2]
    models = []
    for snapshot in (first, later):
        if method == "vi":
            model = StreamingCircularDPMM(
                max_clusters=4, base_mean=means, base_variance=variances
            )
        else:
            model = CollapsedDPSMC(
                snapshot.x, particles=4, base_mean=means, base_variance=variances
            )
        model.partial_fit(snapshot.x, snapshot.theta)
        models.append(model)
    torch.testing.assert_close(models[0].m0, models[1].m0)
    torch.testing.assert_close(models[0].b0, models[1].b0)
    torch.testing.assert_close(
        models[0].m0, torch.tensor(means, dtype=DTYPE, device=DEVICE)
    )


def test_invalid_fixed_base_is_rejected_before_particle_updates():
    first, _ = snapshots((10, 7))
    with pytest.raises(ValueError, match="fixed Normal-Gamma base"):
        CollapsedDPSMC(first.x, particles=4, base_mean=[0, 0], base_variance=[0, 1])
