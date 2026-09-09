"""Unit tests for SABC core algorithm."""

import numpy as np
import pytest

from simulated_annealing_abc import (
    DifferentialEvolution,
    RandomWalk,
    SABCConfig,
    StretchMove,
    make_f_dist,
    sabc,
    update_population,
)
from simulated_annealing_abc.sabc import initialization, update_epsilon_multi_eps


class TestSABCConfig:
    """Tests for SABCConfig validation."""

    def test_init_required_fields(self, mock_f_dist, simple_prior):
        """Test that required fields are set."""
        config = SABCConfig(f_dist=mock_f_dist, prior=simple_prior)
        assert config.f_dist is mock_f_dist
        assert config.prior is simple_prior

    def test_init_defaults(self, mock_f_dist, simple_prior):
        """Test default values."""
        config = SABCConfig(f_dist=mock_f_dist, prior=simple_prior)
        assert config.n_particles == 1000
        assert config.v == 1.0
        assert config.delta == 0.1
        assert config.algorithm == "single_eps"
        assert config.resample is None
        assert config.proposal is None
        assert config.parallel_batches is False
        assert config.annealing_schedule == "curved_geodesic"

    def test_invalid_schedule(self, mock_f_dist, simple_prior):
        """Reject misspelled schedules at configuration time."""
        with pytest.raises(ValueError, match="annealing_schedule"):
            SABCConfig(f_dist=mock_f_dist, prior=simple_prior, annealing_schedule="unknown")

    def test_init_custom_values(self, mock_f_dist, simple_prior, rng):
        """Test custom values."""
        proposal = DifferentialEvolution(n_para=2, rng=rng)
        config = SABCConfig(
            f_dist=mock_f_dist,
            prior=simple_prior,
            n_particles=500,
            v=2.0,
            delta=0.2,
            algorithm="multi_eps",
            resample=100,
            proposal=proposal,
            parallel_batches=True,
            rng=rng,
        )
        assert config.n_particles == 500
        assert config.v == 2.0
        assert config.delta == 0.2
        assert config.algorithm == "multi_eps"
        assert config.resample == 100
        assert config.proposal is proposal
        assert config.parallel_batches is True


class TestEpsilonUpdates:
    """Tests for adaptive temperature updates."""

    def test_multi_epsilon_uses_updated_schedule(self):
        """Test the multi-epsilon update against equations 19 and 20."""
        beta = np.array([1.0, 2.0])
        exp_neg_beta = np.exp(-beta)
        u_bar = (1.0 - exp_neg_beta * (1.0 + beta)) / (beta * (1.0 - exp_neg_beta))
        u = np.tile(u_bar, (8, 1))
        v = 1.7

        cn = 5.0  # (2 * n_stats + 2)! / ((n_stats + 1)! * (n_stats + 2)!)
        beta_effective = beta + v / (cn * u_bar * np.sqrt(np.prod(u_bar)))
        expected = 1.0 / beta_effective

        np.testing.assert_allclose(
            update_epsilon_multi_eps(u, v, "ray_geodesic"), expected, rtol=1e-10
        )

    def test_curved_force_formula(self):
        """Recover the prescribed off-diagonal force with known internal betas."""
        beta = np.array([2.0, 3.0, 4.0])
        mean_u = 1 / beta - 1 / np.expm1(beta)
        delta = np.log(mean_u) - np.log(mean_u).mean()
        rho = np.sqrt(3 / 16 * np.sum(delta**2))
        force = 1.7 / (14 * mean_u * np.sqrt(np.prod(mean_u))) * (
            np.cos(rho) + 3 / 8 * np.sin(rho) / rho * delta
        )
        actual = update_epsilon_multi_eps(np.tile(mean_u, (8, 1)), 1.7)
        np.testing.assert_allclose(actual, 1 / (beta + force), rtol=1e-10)

    @pytest.mark.parametrize("n_stats", [1, 3, 10])
    @pytest.mark.parametrize("perturbation", [0.0, 1e-12])
    def test_curved_diagonal_limit(self, n_stats, perturbation):
        """At and near rho=0 the curved force continuously reduces to the ray."""
        u = np.full((4, n_stats), 0.1)
        u[:, 0] += perturbation
        np.testing.assert_allclose(
            update_epsilon_multi_eps(u, 1.0),
            update_epsilon_multi_eps(u, 1.0, "ray_geodesic"),
            rtol=1e-9,
        )

    @pytest.mark.parametrize("energy", [0.0, 1e-300, 1e-12])
    def test_curved_small_energy(self, energy):
        """Retain positive finite temperatures even when the product underflows."""
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            epsilon = update_epsilon_multi_eps(np.full((2, 100), energy), 1.0)
        assert np.all(np.isfinite(epsilon))
        assert np.all(epsilon > 0)

    def test_curved_allows_negative_external_beta(self):
        """Retain the signed force for strongly anisotropic bounded energies."""
        epsilon = update_epsilon_multi_eps(np.array([[1e-6, 0.4]]), 1.0)
        assert np.all(np.isfinite(epsilon))
        assert np.any(epsilon < 0)

    @pytest.mark.parametrize("schedule", ["ray_geodesic", "curved_geodesic"])
    def test_schedule_used_by_sampler(self, mock_f_dist, simple_prior, schedule):
        """Both initialization and updates use the selected schedule with default DE."""
        config = SABCConfig(
            f_dist=mock_f_dist, prior=simple_prior, n_particles=100,
            algorithm="multi_eps", annealing_schedule=schedule, seed=12,
            show_progressbar=False,
        )
        result = initialization(config, n_simulation=500)
        np.testing.assert_allclose(
            result.state.epsilon,
            update_epsilon_multi_eps(result.u, config.v, schedule),
        )
        result = update_population(result, n_simulation=400)
        assert isinstance(result.config.proposal, DifferentialEvolution)
        np.testing.assert_allclose(
            result.state.epsilon,
            update_epsilon_multi_eps(result.u, config.v, schedule),
        )
        assert result.state.n_population_updates == 4


class TestSABCBasic:
    """Basic SABC unit tests with fast mock simulator."""

    @pytest.fixture
    def fast_config(self, mock_simulator, mock_stats_fn, simple_prior, rng):
        """Create a fast SABCConfig for testing."""
        n_samples = 50
        true_mu, true_sigma = 10.0, 5.0
        y_obs = rng.normal(true_mu, true_sigma, size=n_samples)

        n_stats = 2
        ss_obs = np.empty((1, n_stats), dtype=np.float64)
        mock_stats_fn(y_obs.reshape(1, -1), ss_obs)
        ss_obs = ss_obs.ravel()

        f_dist = make_f_dist(
            n_samples=n_samples,
            ss_obs=ss_obs,
            simulator=mock_simulator,
            stats_fn=mock_stats_fn,
            seed=123,
        )

        return SABCConfig(
            f_dist=f_dist,
            prior=simple_prior,
            n_particles=50,
            v=1.0,
            algorithm="single_eps",
            proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(22)),
            rng=np.random.default_rng(18),
            show_progressbar=False,
        )

    def test_sabc_basic_run(self, fast_config):
        """Test basic SABC run completes."""
        result = sabc(fast_config, n_simulation=1000)

        assert result.population.shape == (50, 2)
        assert np.all(np.isfinite(result.population))
        assert result.state.n_simulation > 0
        assert result.state.n_population_updates > 0

    def test_sabc_multi_eps(self, fast_config):
        """Test SABC with multi_eps algorithm."""
        fast_config.algorithm = "multi_eps"
        result = sabc(fast_config, n_simulation=1000)

        assert result.population.shape == (50, 2)
        assert np.all(np.isfinite(result.population))

    def test_sabc_result_structure(self, fast_config):
        """Test SABCResult has expected fields."""
        result = sabc(fast_config, n_simulation=1000)

        assert hasattr(result, "population")
        assert hasattr(result, "rho")
        assert hasattr(result, "state")
        assert hasattr(result.state, "n_simulation")
        assert hasattr(result.state, "n_population_updates")
        assert hasattr(result.state, "n_accept")
        assert hasattr(result.state, "epsilon_history")
        assert hasattr(result.state, "rho_history")
        assert hasattr(result.state, "u_history")

    def test_sabc_epsilon_decreases(self, fast_config):
        """Test that epsilon decreases during annealing."""
        result = sabc(fast_config, n_simulation=2000)

        eps_hist = result.state.epsilon_history
        assert len(eps_hist) >= 2

        eps_first = np.mean(eps_hist[1])
        eps_last = np.mean(eps_hist[-1])
        assert eps_last < eps_first, "Epsilon should decrease during annealing"

    def test_update_population_continuation(self, fast_config):
        """Test that update_population continues from previous result."""
        result1 = sabc(fast_config, n_simulation=1000)

        n_updates_before = result1.state.n_population_updates

        result2 = update_population(result1, n_simulation=1000)

        assert result2.state.n_population_updates > n_updates_before
        assert result2.population.shape == result1.population.shape

    def test_sabc_reproducibility(self, fast_config, mock_simulator, mock_stats_fn, simple_prior):
        """Test that same seed produces same result."""
        n_samples = 50
        rng_data = np.random.default_rng(42)
        y_obs = rng_data.normal(10.0, 5.0, size=n_samples)

        n_stats = 2
        ss_obs = np.empty((1, n_stats), dtype=np.float64)
        mock_stats_fn(y_obs.reshape(1, -1), ss_obs)
        ss_obs = ss_obs.ravel()

        def make_config(seed):
            f_dist = make_f_dist(
                n_samples=n_samples,
                ss_obs=ss_obs,
                simulator=mock_simulator,
                stats_fn=mock_stats_fn,
                seed=seed,
            )
            return SABCConfig(
                f_dist=f_dist,
                prior=simple_prior,
                n_particles=30,
                v=1.0,
                algorithm="single_eps",
                proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(seed)),
                rng=np.random.default_rng(seed),
                show_progressbar=False,
            )

        result1 = sabc(make_config(123), n_simulation=500)
        result2 = sabc(make_config(123), n_simulation=500)

        assert np.allclose(result1.population, result2.population)


class TestSABCProposals:
    """Tests for SABC with different proposal mechanisms."""

    @pytest.fixture
    def fast_config_base(self, mock_simulator, mock_stats_fn, simple_prior, rng):
        """Create base config for proposal tests."""
        n_samples = 50
        true_mu, true_sigma = 10.0, 5.0
        y_obs = rng.normal(true_mu, true_sigma, size=n_samples)

        n_stats = 2
        ss_obs = np.empty((1, n_stats), dtype=np.float64)
        mock_stats_fn(y_obs.reshape(1, -1), ss_obs)
        ss_obs = ss_obs.ravel()

        f_dist = make_f_dist(
            n_samples=n_samples,
            ss_obs=ss_obs,
            simulator=mock_simulator,
            stats_fn=mock_stats_fn,
            seed=123,
        )

        return {
            "f_dist": f_dist,
            "prior": simple_prior,
            "n_particles": 50,
            "v": 1.0,
            "algorithm": "single_eps",
            "show_progressbar": False,
        }

    def test_sabc_differential_evolution(self, fast_config_base, rng):
        """Test SABC with DifferentialEvolution proposal."""
        config = SABCConfig(
            **fast_config_base,
            proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(22)),
            rng=np.random.default_rng(18),
        )
        result = sabc(config, n_simulation=1000)

        assert result.population.shape == (50, 2)
        assert np.all(np.isfinite(result.population))

    def test_sabc_random_walk(self, fast_config_base, rng):
        """Test SABC with RandomWalk proposal."""
        config = SABCConfig(
            **fast_config_base,
            proposal=RandomWalk(n_para=2, rng=np.random.default_rng(22)),
            rng=np.random.default_rng(18),
        )
        result = sabc(config, n_simulation=1000)

        assert result.population.shape == (50, 2)
        assert np.all(np.isfinite(result.population))

    def test_sabc_stretch_move(self, fast_config_base, rng):
        """Test SABC with StretchMove proposal."""
        config = SABCConfig(
            **fast_config_base,
            proposal=StretchMove(rng=np.random.default_rng(22)),
            rng=np.random.default_rng(18),
        )
        result = sabc(config, n_simulation=1000)

        assert result.population.shape == (50, 2)
        assert np.all(np.isfinite(result.population))


class TestSABCParallel:
    """Tests for parallel features (from tests_parallel.py)."""

    @pytest.fixture
    def parallel_config_base(self, mock_simulator, mock_stats_fn, simple_prior):
        """Create base config for parallel tests with realistic parameters."""
        n_samples = 200
        true_mu, true_sigma = 10.0, 15.0
        rng_data = np.random.default_rng(1822)
        y_obs = rng_data.normal(true_mu, true_sigma, size=n_samples)

        n_stats = 2
        ss_obs = np.empty((1, n_stats), dtype=np.float64)
        mock_stats_fn(y_obs.reshape(1, -1), ss_obs)
        ss_obs = ss_obs.ravel()

        return {
            "n_samples": n_samples,
            "ss_obs": ss_obs,
            "simulator": mock_simulator,
            "stats_fn": mock_stats_fn,
            "prior": simple_prior,
        }

    def _check_result(self, result, n_particles):
        """Validate basic sanity of SABC result."""
        pop = result.population
        assert pop.shape == (n_particles, 2), f"Bad population shape: {pop.shape}"
        assert np.all(np.isfinite(pop)), "Non-finite population values"
        assert result.state.n_simulation > n_particles, "Too few simulations"
        assert result.state.n_population_updates > 0, "No population updates"

        eps_hist = result.state.epsilon_history
        assert len(eps_hist) >= 2, "Too few epsilon history entries"
        eps_first = np.mean(eps_hist[1])
        eps_last = np.mean(eps_hist[-1])
        assert eps_last < eps_first, f"Epsilon did not decrease: {eps_first:.4g} -> {eps_last:.4g}"

    def test_sabc_n_workers(self, parallel_config_base):
        """Test SABC runs correctly with multi-threaded FDist."""
        base = parallel_config_base
        n_particles = 100
        n_simulation = 20000

        f_dist = make_f_dist(
            n_samples=base["n_samples"],
            ss_obs=base["ss_obs"],
            simulator=base["simulator"],
            stats_fn=base["stats_fn"],
            seed=123,
            n_workers=2,
        )

        config = SABCConfig(
            f_dist=f_dist,
            prior=base["prior"],
            n_particles=n_particles,
            v=1.0,
            algorithm="single_eps",
            proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(22)),
            rng=np.random.default_rng(18),
            show_progressbar=False,
        )

        result = sabc(config, n_simulation=n_simulation)
        self._check_result(result, n_particles)

    def test_sabc_parallel_batches_de(self, parallel_config_base):
        """Test SABC with parallel half-batch updates using DE proposal."""
        base = parallel_config_base
        n_particles = 100
        n_simulation = 20000

        f_dist = make_f_dist(
            n_samples=base["n_samples"],
            ss_obs=base["ss_obs"],
            simulator=base["simulator"],
            stats_fn=base["stats_fn"],
            seed=123,
        )

        config = SABCConfig(
            f_dist=f_dist,
            prior=base["prior"],
            n_particles=n_particles,
            v=1.0,
            algorithm="single_eps",
            proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(22)),
            parallel_batches=True,
            rng=np.random.default_rng(18),
            show_progressbar=False,
        )

        result = sabc(config, n_simulation=n_simulation)
        self._check_result(result, n_particles)

    def test_sabc_parallel_batches_rw(self, parallel_config_base):
        """Test SABC with parallel half-batch updates using RW proposal."""
        base = parallel_config_base
        n_particles = 100
        n_simulation = 20000

        f_dist = make_f_dist(
            n_samples=base["n_samples"],
            ss_obs=base["ss_obs"],
            simulator=base["simulator"],
            stats_fn=base["stats_fn"],
            seed=456,
        )

        config = SABCConfig(
            f_dist=f_dist,
            prior=base["prior"],
            n_particles=n_particles,
            v=1.0,
            algorithm="single_eps",
            proposal=RandomWalk(n_para=2, rng=np.random.default_rng(33)),
            parallel_batches=True,
            rng=np.random.default_rng(44),
            show_progressbar=False,
        )

        result = sabc(config, n_simulation=n_simulation)
        self._check_result(result, n_particles)

    def test_sabc_parallel_batches_sm(self, parallel_config_base):
        """Test SABC with parallel half-batch updates using StretchMove proposal."""
        base = parallel_config_base
        n_particles = 100
        n_simulation = 20000

        f_dist = make_f_dist(
            n_samples=base["n_samples"],
            ss_obs=base["ss_obs"],
            simulator=base["simulator"],
            stats_fn=base["stats_fn"],
            seed=789,
        )

        config = SABCConfig(
            f_dist=f_dist,
            prior=base["prior"],
            n_particles=n_particles,
            v=1.0,
            algorithm="single_eps",
            proposal=StretchMove(rng=np.random.default_rng(55)),
            parallel_batches=True,
            rng=np.random.default_rng(66),
            show_progressbar=False,
        )

        result = sabc(config, n_simulation=n_simulation)
        self._check_result(result, n_particles)

    def test_sabc_combined_n_workers_parallel_batches(self, parallel_config_base):
        """Test both n_workers > 1 AND parallel_batches=True simultaneously."""
        base = parallel_config_base
        n_particles = 100
        n_simulation = 20000

        f_dist = make_f_dist(
            n_samples=base["n_samples"],
            ss_obs=base["ss_obs"],
            simulator=base["simulator"],
            stats_fn=base["stats_fn"],
            seed=111,
            n_workers=2,
        )

        config = SABCConfig(
            f_dist=f_dist,
            prior=base["prior"],
            n_particles=n_particles,
            v=1.0,
            algorithm="single_eps",
            proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(77)),
            parallel_batches=True,
            rng=np.random.default_rng(88),
            show_progressbar=False,
        )

        result = sabc(config, n_simulation=n_simulation)
        self._check_result(result, n_particles)

        result2 = update_population(result, n_simulation=n_simulation)
        self._check_result(result2, n_particles)

    def test_sabc_multi_eps_parallel_batches(self, parallel_config_base):
        """Test parallel_batches with multi_eps algorithm."""
        base = parallel_config_base
        n_particles = 100
        n_simulation = 20000

        f_dist = make_f_dist(
            n_samples=base["n_samples"],
            ss_obs=base["ss_obs"],
            simulator=base["simulator"],
            stats_fn=base["stats_fn"],
            seed=222,
        )

        config = SABCConfig(
            f_dist=f_dist,
            prior=base["prior"],
            n_particles=n_particles,
            v=1.0,
            algorithm="multi_eps",
            proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(99)),
            parallel_batches=True,
            rng=np.random.default_rng(100),
            show_progressbar=False,
        )

        result = sabc(config, n_simulation=n_simulation)
        self._check_result(result, n_particles)
