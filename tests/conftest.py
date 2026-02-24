"""Shared pytest fixtures for SimulatedAnnealingABC tests."""

import numpy as np
import pytest

from simulated_annealing_abc import make_f_dist


@pytest.fixture
def rng():
    """Return a seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_prior():
    """Return a simple 2D uniform prior over (mu, sigma)."""

    class Prior:
        def __init__(self, mu_min=-10.0, mu_max=20.0, sigma_min=0.1, sigma_max=25.0):
            self.mu_min = mu_min
            self.mu_max = mu_max
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max

        def rvs(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
            mu = rng.uniform(self.mu_min, self.mu_max, size=size)
            sigma = rng.uniform(self.sigma_min, self.sigma_max, size=size)
            return np.column_stack([mu, sigma])

        def logpdf(self, theta: np.ndarray) -> np.ndarray:
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
            lp[in_bounds] = -np.log(self.mu_max - self.mu_min) - np.log(
                self.sigma_max - self.sigma_min
            )
            return lp

    return Prior()


@pytest.fixture
def mock_simulator():
    """Return a batch simulator function for a Gaussian model."""

    def simulator(theta: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
        mu = theta[:, 0:1]
        sigma = theta[:, 1:2]
        y[:] = rng.normal(loc=mu, scale=sigma, size=y.shape)

    return simulator


@pytest.fixture
def mock_stats_fn():
    """Return a batch summary statistics function (mean, std)."""

    def stats_fn(y: np.ndarray, ss_out: np.ndarray) -> None:
        ss_out[:, 0] = np.mean(y, axis=1)
        ss_out[:, 1] = np.std(y, axis=1, ddof=0)

    return stats_fn


@pytest.fixture
def mock_f_dist(mock_simulator, mock_stats_fn, rng):
    """Return a simple FDist for testing."""
    n_samples = 100
    true_mu, true_sigma = 10.0, 5.0
    y_obs = rng.normal(true_mu, true_sigma, size=n_samples)

    n_stats = 2
    ss_obs = np.empty((1, n_stats), dtype=np.float64)
    mock_stats_fn(y_obs.reshape(1, -1), ss_obs)
    ss_obs = ss_obs.ravel()

    return make_f_dist(
        n_samples=n_samples,
        ss_obs=ss_obs,
        simulator=mock_simulator,
        stats_fn=mock_stats_fn,
        seed=123,
        n_workers=1,
    )


@pytest.fixture
def mock_f_dist_multi_worker(mock_simulator, mock_stats_fn, rng):
    """Return an FDist with n_workers=2 for testing parallelism."""
    n_samples = 100
    true_mu, true_sigma = 10.0, 5.0
    y_obs = rng.normal(true_mu, true_sigma, size=n_samples)

    n_stats = 2
    ss_obs = np.empty((1, n_stats), dtype=np.float64)
    mock_stats_fn(y_obs.reshape(1, -1), ss_obs)
    ss_obs = ss_obs.ravel()

    return make_f_dist(
        n_samples=n_samples,
        ss_obs=ss_obs,
        simulator=mock_simulator,
        stats_fn=mock_stats_fn,
        seed=456,
        n_workers=2,
    )
