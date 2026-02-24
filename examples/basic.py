#!/usr/bin/env python3
"""Run profiling for the SABC implementation."""

import argparse
import logging

import numba as nb
import numpy as np
from rich.logging import RichHandler

from simulated_annealing_abc import (
    DifferentialEvolution,
    SABCConfig,
    make_f_dist,
    sabc,
)

LOG = logging.getLogger(__name__)

FORMAT = "%(message)s"
logging.basicConfig(
    level="DEBUG",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
    force=True,
)

LOG.setLevel(logging.DEBUG)


def init_args() -> dict:
    """Initialize arguments."""
    parser = argparse.ArgumentParser(
        description="Frontmatter helper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--use-numba",
        action="store_true",
        help="Use fast mode, in which the simulator and stats_fn are implemented in Numba.",
    )

    parser.add_argument(
        "--parallel-batch",
        action="store_true",
        help="Calculate batches in parallel",
    )
    parser.add_argument(
        "--fdist-workers",
        type=int,  # convert the argument to int
        required=False,  # optional; omit if you want it mandatory
        default=1,  # fallback when the flag isn’t provided
        help="number of workers for f_dist (default: 1)",
    )

    args = vars(parser.parse_args())
    LOG.info(f"Arguments: {args}")
    return args


class Prior:
    """Uniform prior over (mu, sigma).

    Batch API:
      - ``rvs(rng, size=n_particles)`` → ``(n_particles, 2)``
      - ``logpdf(theta_batch)`` → ``(n_particles,)``  where ``theta_batch`` is ``(n_particles, 2)``
    """

    def __init__(self, mu_min, mu_max, sigma_min, sigma_max):
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def rvs(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw ``size`` samples from the prior, returning shape ``(size, 2)``."""
        mu = rng.uniform(self.mu_min, self.mu_max, size=size)
        sigma = rng.uniform(self.sigma_min, self.sigma_max, size=size)
        return np.column_stack([mu, sigma])

    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate log-prior for a batch of parameters.

        Args:
            theta: Parameter array of shape ``(n_particles, 2)``.

        Returns:
            Log-prior values of shape ``(n_particles,)``.
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
    """Batch simulator.

    theta (n_batch_particles, 2), y (n_batch_particles, n_samples).
    Fills y in-place.
    """
    mu = theta[:, 0:1]  # (n_batch_particles, 1)
    sigma = theta[:, 1:2]  # (n_batch_particles, 1)
    y[:] = rng.normal(loc=mu, scale=sigma, size=y.shape)


def stats_fn(y: np.ndarray, ss_out: np.ndarray) -> None:
    """Batch stats.

    y (n_batch_particles, n_samples), ss_out (n_batch_particles, n_stats).
    Fills ss_out in-place.
    """
    ss_out[:, 0] = np.mean(y, axis=1)
    ss_out[:, 1] = np.std(y, axis=1, ddof=0)


@nb.njit(cache=True)
def simulator_nb(theta, y):
    """Single-particle: theta is 1-D (n_para,), y is 1-D (n_samples,). Fills y in-place."""
    mu = theta[0]
    sigma = theta[1]
    tmp = np.random.normal(0.0, 1.0, y.size)
    for i in range(y.size):
        y[i] = mu + sigma * tmp[i]


@nb.njit(cache=True)
def stats_fn_nb(y, ss):
    """Single-particle: y is 1-D (n_samples,), ss is 1-D (n_stats,). Fills ss in-place."""
    s = 0.0
    for i in range(y.size):
        s += y[i]
    m = s / y.size
    v = 0.0
    for i in range(y.size):
        d = y[i] - m
        v += d * d
    ss[0] = m
    ss[1] = np.sqrt(v / y.size)


if __name__ == "__main__":
    args = init_args()

    true_mu = 10.0
    true_sigma = 15.0
    np.random.seed(1822)
    y_obs = np.random.normal(true_mu, true_sigma, size=1000)

    prior = Prior(mu_min=-10.0, mu_max=20.0, sigma_min=0.0, sigma_max=25.0)

    # Compute observed summary statistics using the batch stats_fn
    n_stats = 2
    ss_obs = np.empty((1, n_stats), dtype=np.float64)
    stats_fn(y_obs.reshape(1, -1), ss_obs)
    ss_obs = ss_obs.ravel()

    if args["use_numba"]:
        f_dist = make_f_dist(
            n_samples=1000,
            ss_obs=ss_obs,
            simulator=simulator_nb,
            stats_fn=stats_fn_nb,
            distance="abs",
            use_numba=True,
            n_workers=args["fdist_workers"],
        )
    else:
        f_dist = make_f_dist(
            n_samples=1000,
            ss_obs=ss_obs,
            simulator=simulator,
            stats_fn=stats_fn,
            seed=123,
            distance="abs",
            n_workers=args["fdist_workers"],
        )

    rng_alg = np.random.default_rng(18)
    rng_prop = np.random.default_rng(22)
    config = SABCConfig(
        f_dist=f_dist,
        prior=prior,
        n_particles=1000,
        v=1.0,
        show_checkpoint=500,
        show_progressbar=True,
        algorithm="single_eps",  # or "multi_eps"
        proposal=DifferentialEvolution(n_para=2, rng=rng_prop),
        rng=rng_alg,
        parallel_batches=args["parallel_batch"],
    )

    result = sabc(config, n_simulation=1_000_000)
    # result_2 = update_population(result, n_simulation=1_000_000)
