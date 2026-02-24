"""Unit tests for FDist and make_f_dist."""

import numpy as np
import pytest

from simulated_annealing_abc import FDist, make_f_dist


class TestFDist:
    """Tests for FDist class."""

    def test_init_default(self, mock_simulator, mock_stats_fn, rng):
        """Test default initialization."""
        n_samples = 100
        ss_obs = np.array([0.0, 1.0])

        fdist = FDist(
            n_samples=n_samples,
            ss_obs=ss_obs,
            simulator=mock_simulator,
            stats_fn=mock_stats_fn,
            seed=42,
        )

        assert fdist.n_samples == n_samples
        assert np.array_equal(fdist.ss_obs, ss_obs)
        assert fdist.distance == "abs"
        assert fdist.n_workers == 1

    def test_init_with_weights(self, mock_simulator, mock_stats_fn):
        """Test initialization with weighted squared distance."""
        n_samples = 100
        ss_obs = np.array([0.0, 1.0])
        weights = np.array([1.0, 2.0])

        fdist = FDist(
            n_samples=n_samples,
            ss_obs=ss_obs,
            simulator=mock_simulator,
            stats_fn=mock_stats_fn,
            distance="weighted_sq",
            weights=weights,
        )

        assert fdist.distance == "weighted_sq"
        assert np.array_equal(fdist.weights, weights)

    def test_call_basic(self, mock_f_dist, simple_prior, rng):
        """Test basic FDist call."""
        theta = simple_prior.rvs(rng, size=10)
        result = mock_f_dist(theta)

        assert result.shape == (10, 2)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_call_with_out_buffer(self, mock_f_dist, simple_prior, rng):
        """Test FDist call with pre-allocated output buffer."""
        theta = simple_prior.rvs(rng, size=10)
        out = np.empty((10, 2), dtype=np.float64)

        result = mock_f_dist(theta, out=out)

        assert result is out
        assert np.all(np.isfinite(out))

    def test_call_out_buffer_wrong_shape(self, mock_f_dist, simple_prior, rng):
        """Test that wrong-shaped buffer raises error."""
        theta = simple_prior.rvs(rng, size=10)
        out = np.empty((5, 2), dtype=np.float64)

        with pytest.raises(ValueError, match="shape"):
            mock_f_dist(theta, out=out)

    def test_distance_modes(self, mock_simulator, mock_stats_fn, rng):
        """Test different distance modes."""
        n_samples = 100
        true_mu, true_sigma = 10.0, 5.0
        y_obs = rng.normal(true_mu, true_sigma, size=n_samples)

        n_stats = 2
        ss_obs = np.empty((1, n_stats), dtype=np.float64)
        mock_stats_fn(y_obs.reshape(1, -1), ss_obs)
        ss_obs = ss_obs.ravel()

        theta = np.array([[true_mu, true_sigma]])

        for mode in ["abs", "sq", "weighted_sq"]:
            weights = np.array([1.0, 1.0]) if mode == "weighted_sq" else None
            fdist = make_f_dist(
                n_samples=n_samples,
                ss_obs=ss_obs,
                simulator=mock_simulator,
                stats_fn=mock_stats_fn,
                seed=123,
                distance=mode,
                weights=weights,
            )
            result = fdist(theta)
            assert result.shape == (1, 2)
            assert np.all(np.isfinite(result))


class TestFDistMultiWorker:
    """Tests for multi-threaded FDist."""

    def test_n_workers_shape(self, mock_f_dist_multi_worker, simple_prior, rng):
        """Test that n_workers > 1 produces correct output shape."""
        theta = simple_prior.rvs(rng, size=50)

        result = mock_f_dist_multi_worker(theta)

        assert result.shape == (50, 2)
        assert np.all(np.isfinite(result))

    def test_n_workers_vs_single(self, mock_simulator, mock_stats_fn, rng):
        """Test that n_workers > 1 gives same shape as n_workers=1."""
        n_samples = 100
        true_mu, true_sigma = 10.0, 5.0
        y_obs = rng.normal(true_mu, true_sigma, size=n_samples)

        n_stats = 2
        ss_obs = np.empty((1, n_stats), dtype=np.float64)
        mock_stats_fn(y_obs.reshape(1, -1), ss_obs)
        ss_obs = ss_obs.ravel()

        f1 = make_f_dist(
            n_samples=n_samples,
            ss_obs=ss_obs,
            simulator=mock_simulator,
            stats_fn=mock_stats_fn,
            seed=42,
            n_workers=1,
        )
        f2 = make_f_dist(
            n_samples=n_samples,
            ss_obs=ss_obs,
            simulator=mock_simulator,
            stats_fn=mock_stats_fn,
            seed=99,
            n_workers=2,
        )

        theta = rng.uniform(-5, 15, size=(50, 2))
        theta[:, 1] = rng.uniform(0.1, 20, size=50)

        r1 = f1(theta)
        r2 = f2(theta)

        assert r1.shape == r2.shape
        assert np.all(np.isfinite(r1))
        assert np.all(np.isfinite(r2))


class TestFDistClone:
    """Tests for FDist.clone() method."""

    def test_clone_basic(self, mock_simulator, mock_stats_fn, rng):
        """Test that clone() creates independent copy."""
        n_samples = 100
        ss_obs = np.array([0.0, 1.0])

        f_orig = FDist(
            n_samples=n_samples,
            ss_obs=ss_obs,
            simulator=mock_simulator,
            stats_fn=mock_stats_fn,
            seed=42,
            n_workers=1,
        )

        f_clone = f_orig.clone(seed=99)

        assert f_clone.n_samples == f_orig.n_samples
        assert np.array_equal(f_clone.ss_obs, f_orig.ss_obs)
        assert f_clone.distance == f_orig.distance
        assert f_clone.seed == 99

    def test_clone_independence(self, mock_simulator, mock_stats_fn, simple_prior, rng):
        """Test that cloned FDist with different seed produces different results."""
        n_samples = 100
        true_mu, true_sigma = 10.0, 5.0
        y_obs = rng.normal(true_mu, true_sigma, size=n_samples)

        n_stats = 2
        ss_obs = np.empty((1, n_stats), dtype=np.float64)
        mock_stats_fn(y_obs.reshape(1, -1), ss_obs)
        ss_obs = ss_obs.ravel()

        f_orig = FDist(
            n_samples=n_samples,
            ss_obs=ss_obs,
            simulator=mock_simulator,
            stats_fn=mock_stats_fn,
            seed=42,
            n_workers=1,
        )
        f_clone = f_orig.clone(seed=99)

        theta = simple_prior.rvs(rng, size=20)

        r_orig = f_orig(theta)
        r_clone = f_clone(theta)

        assert r_orig.shape == r_clone.shape
        assert not np.allclose(r_orig, r_clone), (
            "Clone with different seed should produce different results"
        )

    def test_clone_multi_worker(self, mock_simulator, mock_stats_fn, rng):
        """Test clone() with multi-worker FDist."""
        n_samples = 100
        ss_obs = np.array([0.0, 1.0])

        f_orig = FDist(
            n_samples=n_samples,
            ss_obs=ss_obs,
            simulator=mock_simulator,
            stats_fn=mock_stats_fn,
            seed=42,
            n_workers=2,
        )

        f_clone = f_orig.clone(seed=99)

        assert f_clone.n_workers == f_orig.n_workers


class TestMakeFDist:
    """Tests for make_f_dist factory function."""

    def test_make_f_dist_default(self, mock_simulator, mock_stats_fn):
        """Test make_f_dist with default parameters."""
        n_samples = 100
        ss_obs = np.array([0.0, 1.0])

        fdist = make_f_dist(
            n_samples=n_samples,
            ss_obs=ss_obs,
            simulator=mock_simulator,
            stats_fn=mock_stats_fn,
            seed=42,
        )

        assert isinstance(fdist, FDist)
        assert fdist.n_samples == n_samples
        assert fdist.distance == "abs"
        assert fdist.n_workers == 1
