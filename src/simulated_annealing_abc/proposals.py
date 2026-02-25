"""proposals.py — batch proposal generators for SABC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# -------------------------------------------------------
# Base proposal type
# -------------------------------------------------------
class Proposal:
    """Base class for proposal generators.

    All proposals operate in batch mode:

    ``proposal(theta_batch, pop_inactive) -> (theta_prop_batch, log_factors)``

    where ``theta_batch`` has shape ``(n_batch_particles, n_para)`` and ``pop_inactive`` has
    shape ``(n_inactive, n_para)``.  Returns ``(n_batch_particles, n_para)`` proposals and
    ``(n_batch_particles,)`` log Metropolis-Hastings correction factors.
    """

    def update(self, population: np.ndarray) -> None:
        """Recompute internal state from the current population (optional)."""


# -------------------------------------------------------
# Random Walk proposal
# -------------------------------------------------------
@dataclass(init=False)
class RandomWalk(Proposal):
    """Gaussian random walk proposal (batch mode).

    Parameters
    ----------
    beta : float
        Mixing parameter in (0, 1].
    Sigma : float or np.ndarray
        Jump variance (1D) or covariance matrix (nD), adapted from population.
    """

    beta: float
    Sigma: float | np.ndarray  # scalar variance or covariance matrix
    rng: np.random.Generator

    def __init__(self, *, beta: float = 0.8, n_para: int = 1, rng=None):
        if not (0.0 < beta <= 1.0):
            raise ValueError("Mixing parameter `beta` must be between 0 and 1.")
        self.beta = float(beta)
        self.rng = np.random.default_rng() if rng is None else rng

        if n_para == 1:
            self.Sigma = -1.0
        else:
            self.Sigma = -np.ones((n_para, n_para), dtype=float)

    def __call__(
        self,
        theta: np.ndarray,
        population: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propose a batch of new particles via Gaussian random walk.

        Args:
            theta: Current particle positions, shape ``(n_batch_particles, n_para)``.
            population: Inactive population (unused by RandomWalk, kept for interface).

        Returns:
            Tuple of proposed positions ``(n_batch_particles, n_para)`` and log factors
            ``(n_batch_particles,)`` (always zero).
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        n_batch_particles, n_para = theta.shape
        rng = self.rng

        # 1D case: Sigma is a scalar variance
        if np.isscalar(self.Sigma):
            var = float(self.Sigma)
            if var <= 0.0:
                raise RuntimeError("RandomWalk Sigma not updated yet.")
            steps = rng.normal(0.0, np.sqrt(var), size=n_batch_particles)
            return theta + steps[:, np.newaxis], np.zeros(n_batch_particles)

        # nD case: Sigma is a covariance matrix
        cov = np.asarray(self.Sigma, dtype=float)
        if cov.ndim != 2:
            raise ValueError("RandomWalk Sigma must be a covariance matrix.")
        if n_para != cov.shape[0]:
            raise ValueError("theta dimension does not match Sigma.")
        steps = rng.multivariate_normal(np.zeros(n_para), cov, size=n_batch_particles)
        return theta + steps, np.zeros(n_batch_particles)

    def update(self, population: np.ndarray) -> None:
        """Recompute jump covariance from the current population."""
        pop = np.asarray(population, dtype=float)
        if pop.ndim != 2:
            raise ValueError("population must be a 2D array (n_particles, n_para).")

        # 1D case
        if np.isscalar(self.Sigma):
            if pop.shape[1] != 1:
                raise ValueError("RandomWalk initialized for 1D but population is multi-D.")
            var = float(np.var(pop[:, 0], ddof=1))
            self.Sigma = max(self.beta * var, 1e-12)
            return

        # nD case
        cov = np.cov(pop, rowvar=False, bias=False)
        n_para = cov.shape[0]
        self.Sigma = self.beta * (cov + 1e-8 * np.eye(n_para))

    def clone(self, rng: np.random.Generator) -> RandomWalk:
        """Create an independent copy with a different RNG.

        Copies the current ``Sigma`` so the clone starts with the same
        adapted jump distribution.

        Args:
            rng: Random number generator for the new instance.

        Returns:
            A new ``RandomWalk`` with the same config and ``Sigma`` but its own RNG.
        """
        n_para = 1 if np.isscalar(self.Sigma) else self.Sigma.shape[0]
        new = RandomWalk(beta=self.beta, n_para=n_para, rng=rng)
        if np.isscalar(self.Sigma):
            new.Sigma = self.Sigma
        else:
            new.Sigma = self.Sigma.copy()
        return new


# -------------------------------------------------------
# Differential Evolution proposal
# -------------------------------------------------------
@dataclass(init=False)
class DifferentialEvolution(Proposal):
    """Differential Evolution proposal (batch mode).

    If ``n_para`` is given, uses ``gamma0 = 2.38 / sqrt(2 * n_para)``,
    matching the typical DE scaling used in ensemble samplers.
    """

    gamma0: float
    sigma_gamma: float
    rng: np.random.Generator

    def __init__(
        self,
        *,
        gamma0: float | None = None,
        n_para: int | None = None,
        sigma_gamma: float = 1e-5,
        rng=None,
    ):
        if (gamma0 is None) == (n_para is None):
            raise ValueError("Provide exactly one of `gamma0` or `n_para`.")
        if gamma0 is None:
            gamma0 = 2.38 / np.sqrt(2.0 * float(n_para))  # type: ignore[arg-type]
        self.gamma0 = float(gamma0)
        self.sigma_gamma = float(sigma_gamma)
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,
        theta: np.ndarray,
        population: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propose a batch of new particles via Differential Evolution.

        Args:
            theta: Current particle positions, shape ``(n_batch_particles, n_para)``.
            population: Inactive half of the population, shape ``(n_inactive, n_para)``.

        Returns:
            Tuple of proposed positions ``(n_batch_particles, n_para)`` and log factors
            ``(n_batch_particles,)`` (always zero).
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        n_batch_particles = theta.shape[0]
        pop = population
        if pop.ndim != 2:
            raise ValueError("population must be 2D (n_inactive, n_para).")
        n_inactive = pop.shape[0]
        if n_inactive < 2:
            raise ValueError("Population must contain at least 2 particles.")

        rng = self.rng

        # Pick two distinct partners for every particle in the batch
        i1 = rng.integers(n_inactive, size=n_batch_particles)
        i2 = rng.integers(n_inactive - 1, size=n_batch_particles)
        i2[i2 >= i1] += 1

        gamma = self.gamma0 * (1.0 + self.sigma_gamma * rng.standard_normal(n_batch_particles))
        proposal = theta + gamma[:, np.newaxis] * (pop[i1] - pop[i2])
        return proposal, np.zeros(n_batch_particles)

    def clone(self, rng: np.random.Generator) -> DifferentialEvolution:
        """Create an independent copy with a different RNG.

        Args:
            rng: Random number generator for the new instance.

        Returns:
            A new ``DifferentialEvolution`` with the same config but its own RNG.
        """
        return DifferentialEvolution(gamma0=self.gamma0, sigma_gamma=self.sigma_gamma, rng=rng)


# -------------------------------------------------------
# Stretch Move proposal
# -------------------------------------------------------
@dataclass(init=False)
class StretchMove(Proposal):
    """Stretch move proposal (Goodman & Weare / emcee-style, batch mode)."""

    a: float
    rng: np.random.Generator

    def __init__(self, *, a: float = 2.0, rng=None):
        if a <= 1.0:
            raise ValueError("StretchMove parameter 'a' must be > 1.")
        self.a = float(a)
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,
        theta: np.ndarray,
        population: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propose a batch of new particles via the stretch move.

        Args:
            theta: Current particle positions, shape ``(n_batch_particles, n_para)``.
            population: Inactive half of the population, shape ``(n_inactive, n_para)``.

        Returns:
            Tuple of proposed positions ``(n_batch_particles, n_para)`` and log factors
            ``(n_batch_particles,)``.
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        n_batch_particles, n_para = theta.shape
        pop = population
        if pop.ndim != 2:
            raise ValueError("population must be 2D (n_inactive, n_para).")
        n_inactive = pop.shape[0]
        if n_inactive < 1:
            raise ValueError("Population must not be empty.")

        rng = self.rng
        i = rng.integers(n_inactive, size=n_batch_particles)
        partners = pop[i, :]

        u_rand = rng.random(n_batch_particles)
        z = ((self.a - 1.0) * u_rand + 1.0) ** 2 / self.a

        log_factors = np.log(z) * (n_para - 1)
        proposal = partners + z[:, np.newaxis] * (theta - partners)
        return proposal, log_factors

    def clone(self, rng: np.random.Generator) -> StretchMove:
        """Create an independent copy with a different RNG.

        Args:
            rng: Random number generator for the new instance.

        Returns:
            A new ``StretchMove`` with the same config but its own RNG.
        """
        return StretchMove(a=self.a, rng=rng)
