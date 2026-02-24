#!/usr/bin/env python3
"""Smoke tests for parallelism features: n_workers and parallel_batches.

Tests that both n_workers > 1 (multi-threaded FDist) and parallel_batches=True
(concurrent half-batch updates) run without errors and produce reasonable output.
Uses small n_particles and n_simulation for speed.
"""

import logging
import time

import numpy as np

from simulated_annealing_abc import (
    DifferentialEvolution,
    FDist,
    RandomWalk,
    SABCConfig,
    StretchMove,
    make_f_dist,
    sabc,
    update_population,
)

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)


# ======================================================================
# Shared test fixtures
# ======================================================================
TRUE_MU = 10.0
TRUE_SIGMA = 15.0
N_SAMPLES = 200

rng_data = np.random.default_rng(1822)
Y_OBS = rng_data.normal(TRUE_MU, TRUE_SIGMA, size=N_SAMPLES)


class Prior:
    """Uniform prior over (mu, sigma)."""

    def __init__(self, mu_min=-10.0, mu_max=20.0, sigma_min=0.1, sigma_max=25.0):
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def rvs(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw samples from the prior."""
        mu = rng.uniform(self.mu_min, self.mu_max, size=size)
        sigma = rng.uniform(self.sigma_min, self.sigma_max, size=size)
        return np.column_stack([mu, sigma])

    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate log-prior for a batch of parameters."""
        theta = np.atleast_2d(theta)
        mu = theta[:, 0]
        sigma = theta[:, 1]
        in_bounds = (
            (self.mu_min <= mu)
            & (mu <= self.mu_max)
            & (self.sigma_min <= sigma)
            & (sigma <= self.sigma_max)
        )
        lp = np.full(theta.shape[0], -np.inf)
        lp[in_bounds] = -np.log(self.mu_max - self.mu_min) - np.log(self.sigma_max - self.sigma_min)
        return lp


def simulator(theta: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
    """Batch simulator: fills y in-place."""
    mu = theta[:, 0:1]
    sigma = theta[:, 1:2]
    y[:] = rng.normal(loc=mu, scale=sigma, size=y.shape)


def stats_fn(y: np.ndarray, ss_out: np.ndarray) -> None:
    """Batch stats: fills ss_out in-place."""
    ss_out[:, 0] = np.mean(y, axis=1)
    ss_out[:, 1] = np.std(y, axis=1, ddof=0)


PRIOR = Prior()
N_STATS = 2
SS_OBS = np.empty((1, N_STATS), dtype=np.float64)
stats_fn(Y_OBS.reshape(1, -1), SS_OBS)
SS_OBS = SS_OBS.ravel()

# Small parameters for fast smoke tests
N_PARTICLES = 100
N_SIMULATION = 20_000


def _check_result(result, label: str) -> None:
    """Validate basic sanity of SABC result."""
    pop = result.population
    assert pop.shape == (N_PARTICLES, 2), f"[{label}] Bad population shape: {pop.shape}"
    assert np.all(np.isfinite(pop)), f"[{label}] Non-finite population values"
    assert result.state.n_simulation > N_PARTICLES, f"[{label}] Too few simulations"
    assert result.state.n_population_updates > 0, f"[{label}] No population updates"

    # Check epsilon decreased (annealing progressed)
    eps_hist = result.state.epsilon_history
    assert len(eps_hist) >= 2, f"[{label}] Too few epsilon history entries"
    eps_first = np.mean(eps_hist[1])  # skip init
    eps_last = np.mean(eps_hist[-1])
    assert eps_last < eps_first, (
        f"[{label}] Epsilon did not decrease: {eps_first:.4g} -> {eps_last:.4g}"
    )
    LOG.info(
        f"[{label}] OK: {result.state.n_population_updates} updates, "
        f"eps {eps_first:.4g} -> {eps_last:.4g}, "
        f"n_accept={result.state.n_accept}"
    )


# ======================================================================
# Test 1: FDist with n_workers > 1
# ======================================================================
def test_fdist_n_workers():
    """Test that FDist produces identical-shape output with multiple workers."""
    LOG.info("=" * 60)
    LOG.info("TEST: FDist n_workers")

    f1 = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=SS_OBS,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=42,
        n_workers=1,
    )
    f2 = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=SS_OBS,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=99,
        n_workers=2,
    )

    theta_test = PRIOR.rvs(np.random.default_rng(0), size=50)

    r1 = f1(theta_test)
    r2 = f2(theta_test)

    assert r1.shape == r2.shape == (50, 2), f"Shape mismatch: {r1.shape} vs {r2.shape}"
    assert np.all(np.isfinite(r1)), "Non-finite values from n_workers=1"
    assert np.all(np.isfinite(r2)), "Non-finite values from n_workers=2"
    LOG.info("FDist n_workers: shapes and finiteness OK")


# ======================================================================
# Test 2: FDist.clone()
# ======================================================================
def test_fdist_clone():
    """Test that FDist.clone() creates an independent copy."""
    LOG.info("=" * 60)
    LOG.info("TEST: FDist.clone()")

    f_orig = FDist(
        n_samples=N_SAMPLES,
        ss_obs=SS_OBS,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=42,
        n_workers=2,
    )
    f_clone = f_orig.clone(seed=99)

    theta_test = PRIOR.rvs(np.random.default_rng(0), size=20)

    r_orig = f_orig(theta_test)
    r_clone = f_clone(theta_test)

    assert r_orig.shape == r_clone.shape, f"Shape mismatch: {r_orig.shape} vs {r_clone.shape}"
    # Different seeds -> different RNG -> different results (stochastic simulator)
    assert not np.allclose(r_orig, r_clone), "Clone with different seed produced identical results"
    LOG.info("FDist.clone(): shapes and independence OK")


# ======================================================================
# Test 3: Proposal.clone()
# ======================================================================
def test_proposal_clone():
    """Test that all proposal types implement clone() correctly."""
    LOG.info("=" * 60)
    LOG.info("TEST: Proposal.clone()")

    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(20)

    # RandomWalk
    rw = RandomWalk(n_para=2, rng=rng1)
    pop = PRIOR.rvs(np.random.default_rng(0), size=50)
    rw.update(pop)
    rw_clone = rw.clone(rng2)
    assert rw_clone.beta == rw.beta
    assert np.allclose(rw_clone.Sigma, rw.Sigma)
    assert rw_clone.rng is not rw.rng
    LOG.info("  RandomWalk.clone() OK")

    # DifferentialEvolution
    de = DifferentialEvolution(n_para=2, rng=rng1)
    de_clone = de.clone(rng2)
    assert de_clone.gamma0 == de.gamma0
    assert de_clone.sigma_gamma == de.sigma_gamma
    assert de_clone.rng is not de.rng
    LOG.info("  DifferentialEvolution.clone() OK")

    # StretchMove
    sm = StretchMove(rng=rng1)
    sm_clone = sm.clone(rng2)
    assert sm_clone.a == sm.a
    assert sm_clone.rng is not sm.rng
    LOG.info("  StretchMove.clone() OK")


# ======================================================================
# Test 4: SABC with n_workers > 1 (serial batches)
# ======================================================================
def test_sabc_n_workers():
    """Test SABC runs correctly with multi-threaded FDist."""
    LOG.info("=" * 60)
    LOG.info("TEST: SABC with n_workers=2")

    f_dist = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=SS_OBS,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=123,
        n_workers=2,
    )

    config = SABCConfig(
        f_dist=f_dist,
        prior=PRIOR,
        n_particles=N_PARTICLES,
        v=1.0,
        algorithm="single_eps",
        proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(22)),
        rng=np.random.default_rng(18),
    )

    result = sabc(config, n_simulation=N_SIMULATION)
    _check_result(result, "n_workers=2, serial batches")


# ======================================================================
# Test 5: SABC with parallel_batches=True (DE proposal)
# ======================================================================
def test_sabc_parallel_batches_de():
    """Test SABC with parallel half-batch updates using DE proposal."""
    LOG.info("=" * 60)
    LOG.info("TEST: SABC parallel_batches=True (DE)")

    f_dist = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=SS_OBS,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=123,
    )

    config = SABCConfig(
        f_dist=f_dist,
        prior=PRIOR,
        n_particles=N_PARTICLES,
        v=1.0,
        algorithm="single_eps",
        proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(22)),
        parallel_batches=True,
        rng=np.random.default_rng(18),
    )

    result = sabc(config, n_simulation=N_SIMULATION)
    _check_result(result, "parallel_batches=True, DE")


# ======================================================================
# Test 6: SABC with parallel_batches=True (RW proposal)
# ======================================================================
def test_sabc_parallel_batches_rw():
    """Test SABC with parallel half-batch updates using RW proposal."""
    LOG.info("=" * 60)
    LOG.info("TEST: SABC parallel_batches=True (RW)")

    f_dist = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=SS_OBS,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=456,
    )

    config = SABCConfig(
        f_dist=f_dist,
        prior=PRIOR,
        n_particles=N_PARTICLES,
        v=1.0,
        algorithm="single_eps",
        proposal=RandomWalk(n_para=2, rng=np.random.default_rng(33)),
        parallel_batches=True,
        rng=np.random.default_rng(44),
    )

    result = sabc(config, n_simulation=N_SIMULATION)
    _check_result(result, "parallel_batches=True, RW")


# ======================================================================
# Test 7: SABC with parallel_batches=True (StretchMove)
# ======================================================================
def test_sabc_parallel_batches_sm():
    """Test SABC with parallel half-batch updates using StretchMove proposal."""
    LOG.info("=" * 60)
    LOG.info("TEST: SABC parallel_batches=True (SM)")

    f_dist = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=SS_OBS,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=789,
    )

    config = SABCConfig(
        f_dist=f_dist,
        prior=PRIOR,
        n_particles=N_PARTICLES,
        v=1.0,
        algorithm="single_eps",
        proposal=StretchMove(rng=np.random.default_rng(55)),
        parallel_batches=True,
        rng=np.random.default_rng(66),
    )

    result = sabc(config, n_simulation=N_SIMULATION)
    _check_result(result, "parallel_batches=True, SM")


# ======================================================================
# Test 8: Combined n_workers + parallel_batches
# ======================================================================
def test_sabc_combined():
    """Test both n_workers > 1 AND parallel_batches=True simultaneously."""
    LOG.info("=" * 60)
    LOG.info("TEST: SABC n_workers=2 + parallel_batches=True")

    f_dist = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=SS_OBS,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=111,
        n_workers=2,
    )

    config = SABCConfig(
        f_dist=f_dist,
        prior=PRIOR,
        n_particles=N_PARTICLES,
        v=1.0,
        algorithm="single_eps",
        proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(77)),
        parallel_batches=True,
        rng=np.random.default_rng(88),
    )

    result = sabc(config, n_simulation=N_SIMULATION)
    _check_result(result, "n_workers=2 + parallel_batches=True")

    # Also test update_population continuation
    result2 = update_population(result, n_simulation=N_SIMULATION)
    _check_result(result2, "n_workers=2 + parallel_batches=True (continued)")


# ======================================================================
# Test 9: multi_eps + parallel_batches
# ======================================================================
def test_sabc_multi_eps_parallel():
    """Test parallel_batches with multi_eps algorithm."""
    LOG.info("=" * 60)
    LOG.info("TEST: SABC multi_eps + parallel_batches=True")

    f_dist = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=SS_OBS,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=222,
    )

    config = SABCConfig(
        f_dist=f_dist,
        prior=PRIOR,
        n_particles=N_PARTICLES,
        v=1.0,
        algorithm="multi_eps",
        proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(99)),
        parallel_batches=True,
        rng=np.random.default_rng(100),
    )

    result = sabc(config, n_simulation=N_SIMULATION)
    _check_result(result, "multi_eps + parallel_batches=True")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    t0 = time.perf_counter()

    test_fdist_n_workers()
    test_fdist_clone()
    test_proposal_clone()
    test_sabc_n_workers()
    test_sabc_parallel_batches_de()
    test_sabc_parallel_batches_rw()
    test_sabc_parallel_batches_sm()
    test_sabc_combined()
    test_sabc_multi_eps_parallel()

    elapsed = time.perf_counter() - t0
    LOG.info("=" * 60)
    LOG.info(f"ALL SMOKE TESTS PASSED in {elapsed:.1f}s")
