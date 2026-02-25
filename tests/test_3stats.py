"""Integration tests for SABC with 3 summary statistics.

This test uses empirical mean, median, and standard deviation as summary statistics.
The median is a redundant statistic for a Gaussian model.

Run with visualization:
    VISUALIZE=1 pytest -m slow tests/test_3stats.py
"""

import os
from pathlib import Path

import numpy as np
import pytest

from simulated_annealing_abc import (
    DifferentialEvolution,
    RandomWalk,
    SABCConfig,
    StretchMove,
    make_f_dist,
    sabc,
    save_sabc_result,
)

VISUALIZE = os.environ.get("VISUALIZE", "0") == "1"
RESULTS_DIR = Path(__file__).parent / "test_results"

TRUE_MU = 10.0
TRUE_SIGMA = 15.0
N_SAMPLES = 1000
N_PARTICLES = 1000
N_SIMULATION = 1_000_000


class Prior:
    """Uniform prior over (mu, sigma)."""

    def __init__(self, mu_min=-10.0, mu_max=20.0, sigma_min=0.0, sigma_max=25.0):
        """Initialize prior bounds.

        Args:
            mu_min: Lower bound for mu.
            mu_max: Upper bound for mu.
            sigma_min: Lower bound for sigma.
            sigma_max: Upper bound for sigma.
        """
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def rvs(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw samples from the prior.

        Args:
            rng: Random number generator.
            size: Number of samples.

        Returns:
            Array of shape (size, 2) with (mu, sigma) samples.
        """
        mu = rng.uniform(self.mu_min, self.mu_max, size=size)
        sigma = rng.uniform(self.sigma_min, self.sigma_max, size=size)
        return np.column_stack([mu, sigma])

    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate log-prior density.

        Args:
            theta: Parameter array of shape (n, 2).

        Returns:
            Log-prior values of shape (n,).
        """
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
    mu = theta[:, 0:1]
    sigma = theta[:, 1:2]
    y[:] = rng.normal(loc=mu, scale=sigma, size=y.shape)


def stats_fn_3(y: np.ndarray, ss_out: np.ndarray) -> None:
    """Summary statistics: mean, median, std."""
    ss_out[:, 0] = np.mean(y, axis=1)
    ss_out[:, 1] = np.median(y, axis=1)
    ss_out[:, 2] = np.std(y, axis=1, ddof=0)


@pytest.fixture(scope="module")
def observed_data():
    """Generate observed data once for all tests."""
    rng = np.random.default_rng(1822)
    y_obs = rng.normal(TRUE_MU, TRUE_SIGMA, size=N_SAMPLES)

    n_stats = 3
    ss_obs = np.empty((1, n_stats), dtype=np.float64)
    stats_fn_3(y_obs.reshape(1, -1), ss_obs)
    ss_obs = ss_obs.ravel()

    return y_obs, ss_obs


@pytest.fixture(scope="module")
def prior():
    return Prior()


def _check_posterior(result, true_mu=TRUE_MU, true_sigma=TRUE_SIGMA):
    """Basic posterior sanity check."""
    pop = result.population
    mu_mean = np.mean(pop[:, 0])
    sigma_mean = np.mean(pop[:, 1])

    assert np.abs(mu_mean - true_mu) < 3.0, f"Mean mu {mu_mean:.2f} too far from {true_mu}"
    assert np.abs(sigma_mean - true_sigma) < 3.0, (
        f"Mean sigma {sigma_mean:.2f} too far from {true_sigma}"
    )

    return mu_mean, sigma_mean


def _plot_posterior(result, label, true_mu=TRUE_MU, true_sigma=TRUE_SIGMA):
    """Plot posterior if VISUALIZE=1."""
    if not VISUALIZE:
        return

    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    pop = result.population
    values = np.vstack([pop[:, 0], pop[:, 1]])
    kde = gaussian_kde(values)

    mu_lims = (true_mu - 5, true_mu + 5)
    sigma_lims = (true_sigma - 5, true_sigma + 5)
    mu_grid = np.linspace(mu_lims[0], mu_lims[1], 250)
    sig_grid = np.linspace(sigma_lims[0], sigma_lims[1], 250)
    MU, SIG = np.meshgrid(mu_grid, sig_grid)

    positions = np.vstack([MU.ravel(), SIG.ravel()])
    Z = kde(positions).reshape(MU.shape)

    plt.figure(figsize=(6, 5))
    plt.grid(True, alpha=0.3, zorder=0)
    n_levels = 10
    cf = plt.contourf(MU, SIG, Z, levels=n_levels, cmap="Greys", zorder=1)
    plt.contour(MU, SIG, Z, levels=n_levels, colors="black", linewidths=0.6, alpha=0.6, zorder=2)
    plt.scatter(
        true_mu, true_sigma, c="red", s=90, linewidths=3.0, marker="x", zorder=3, label="True"
    )
    plt.xlim(mu_lims)
    plt.ylim(sigma_lims)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\sigma$")
    plt.title(f"SABC posterior (3 stats): {label}")
    plt.colorbar(cf, label="Posterior density")
    plt.legend()
    plt.show()


@pytest.mark.slow
@pytest.mark.integration
def test_3stats_de(observed_data, prior):
    """Test SABC with 3 stats using DifferentialEvolution proposal."""
    y_obs, ss_obs = observed_data

    f_dist = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=ss_obs,
        simulator=simulator,
        stats_fn=stats_fn_3,
        seed=123,
    )

    config = SABCConfig(
        f_dist=f_dist,
        prior=prior,
        n_particles=N_PARTICLES,
        v=1.0,
        algorithm="single_eps",
        proposal=DifferentialEvolution(n_para=2, rng=np.random.default_rng(22)),
        rng=np.random.default_rng(18),
        show_checkpoint=100000,
    )

    result = sabc(config, n_simulation=N_SIMULATION)
    mu_mean, sigma_mean = _check_posterior(result)
    _plot_posterior(result, "DE")

    RESULTS_DIR.mkdir(exist_ok=True)
    save_sabc_result(result, RESULTS_DIR / "test_3stats_de.pkl")

    print(f"\n3stats DE: mu={mu_mean:.2f}, sigma={sigma_mean:.2f}")


@pytest.mark.slow
@pytest.mark.integration
def test_3stats_rw(observed_data, prior):
    """Test SABC with 3 stats using RandomWalk proposal."""
    y_obs, ss_obs = observed_data

    f_dist = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=ss_obs,
        simulator=simulator,
        stats_fn=stats_fn_3,
        seed=456,
    )

    config = SABCConfig(
        f_dist=f_dist,
        prior=prior,
        n_particles=N_PARTICLES,
        v=1.0,
        algorithm="single_eps",
        proposal=RandomWalk(n_para=2, rng=np.random.default_rng(33)),
        rng=np.random.default_rng(44),
        show_checkpoint=100000,
    )

    result = sabc(config, n_simulation=N_SIMULATION)
    mu_mean, sigma_mean = _check_posterior(result)
    _plot_posterior(result, "RW")

    RESULTS_DIR.mkdir(exist_ok=True)
    save_sabc_result(result, RESULTS_DIR / "test_3stats_rw.pkl")

    print(f"\n3stats RW: mu={mu_mean:.2f}, sigma={sigma_mean:.2f}")


@pytest.mark.slow
@pytest.mark.integration
def test_3stats_sm(observed_data, prior):
    """Test SABC with 3 stats using StretchMove proposal."""
    y_obs, ss_obs = observed_data

    f_dist = make_f_dist(
        n_samples=N_SAMPLES,
        ss_obs=ss_obs,
        simulator=simulator,
        stats_fn=stats_fn_3,
        seed=789,
    )

    config = SABCConfig(
        f_dist=f_dist,
        prior=prior,
        n_particles=N_PARTICLES,
        v=1.0,
        algorithm="single_eps",
        proposal=StretchMove(rng=np.random.default_rng(55)),
        rng=np.random.default_rng(66),
        show_checkpoint=100000,
    )

    result = sabc(config, n_simulation=N_SIMULATION)
    mu_mean, sigma_mean = _check_posterior(result)
    _plot_posterior(result, "SM")

    RESULTS_DIR.mkdir(exist_ok=True)
    save_sabc_result(result, RESULTS_DIR / "test_3stats_sm.pkl")

    print(f"\n3stats SM: mu={mu_mean:.2f}, sigma={sigma_mean:.2f}")
