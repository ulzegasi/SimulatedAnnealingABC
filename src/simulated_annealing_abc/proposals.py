# proposals.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np


# -------------------------------------------------------
# Base proposal type + dispatcher
# -------------------------------------------------------


class Proposal:
    """Base class for proposal generators."""

    def update(self, population: List[Any]) -> None:
        """Update internal state (e.g. covariance)."""
        return


def update_proposal(proposal: Proposal, population: List[Any]) -> None:
    """
    Update the proposal's internal state (e.g. jump covariance).

    This matches the Julia `update_proposal!` interface.
    """
    proposal.update(population)


# -------------------------------------------------------
# Random Walk proposal
# -------------------------------------------------------


@dataclass
class RandomWalk(Proposal):
    """
    Gaussian random walk proposal.

    Parameters
    ----------
    beta : float
        Mixing parameter in (0, 1].
    Sigma : float or np.ndarray
        Jump variance (1D) or covariance matrix (nD).
        Will be adapted from the population.
    """

    beta: float
    Sigma: Any  # float (1D) or np.ndarray (nD)

    def __init__(self, *, beta: float = 0.8, n_para: int):
        if not (0.0 < beta <= 1.0):
            raise ValueError("Mixing parameter `beta` must be between 0 and 1.")
        self.beta = float(beta)
        if n_para == 1:
            # 1D, we store a scalar variance (will be updated)
            self.Sigma = -1.0
        else:
            # nD, store a covariance matrix (will be updated)
            self.Sigma = -np.ones((n_para, n_para))

    def __call__(self, theta: Any, population: List[Any]) -> tuple[Any, float]:
        """
        Generate a proposal from current theta.

        Returns
        -------
        theta_proposal, log_factor
        """
        log_factor = 0.0

        # 1D case: Sigma is scalar
        if np.isscalar(self.Sigma):
            sigma = float(self.Sigma)
            if sigma <= 0:
                raise RuntimeError("RandomWalk Sigma not updated yet.")
            step = np.random.normal(loc=0.0, scale=np.sqrt(sigma))
            return theta + step, log_factor

        # nD case: Sigma is covariance matrix
        cov = np.asarray(self.Sigma, dtype=float)
        if cov.ndim != 2:
            raise ValueError("RandomWalk Sigma must be a covariance matrix.")
        d = cov.shape[0]
        step = np.random.multivariate_normal(mean=np.zeros(d), cov=cov)
        theta_arr = np.asarray(theta, dtype=float)
        return theta_arr + step, log_factor

    def update(self, population: List[Any]) -> None:
        """
        Update covariance (or variance) of the jump distribution from population.
        """
        # 1D case
        if np.isscalar(self.Sigma):
            arr = np.array(
                [np.atleast_1d(p)[0] for p in population],
                dtype=float,
            )
            # unbiased covariance; for 1D this is just variance
            cov = np.cov(arr, bias=False)
            self.Sigma = self.beta * cov
        else:
            # nD case: stack as rows, covariance over columns
            arr = np.stack(
                [np.asarray(p, dtype=float) for p in population],
                axis=0,
            )
            cov = np.cov(arr, rowvar=False, bias=False)
            d = cov.shape[0]
            self.Sigma = self.beta * (cov + 1e-8 * np.eye(d))


# -------------------------------------------------------
# Differential Evolution proposal
# -------------------------------------------------------


@dataclass
class DifferentialEvolution(Proposal):
    """
    Differential Evolution proposal.

    Parameters
    ----------
    gamma0 : float
        Base scale parameter γ₀.
    sigma_gamma : float
        Noise on γ (multiplicative factor).

    If `n_para` is provided instead of `gamma0`,
    we use γ₀ = 2.38 / sqrt(2 * n_para),
    matching the Julia implementation.
    """

    gamma0: float
    sigma_gamma: float = 1e-5

    def __init__(
        self,
        *,
        gamma0: float | None = None,
        n_para: int | None = None,
        sigma_gamma: float = 1e-5,
    ):
        if (gamma0 is None) == (n_para is None):
            raise ValueError("Provide exactly one of `gamma0` or `n_para`.")
        if gamma0 is None:
            gamma0 = 2.38 / np.sqrt(2.0 * n_para)
        self.gamma0 = float(gamma0)
        self.sigma_gamma = float(sigma_gamma)

    def __call__(self, theta: Any, population: List[Any]) -> tuple[Any, float]:
        """
        Propose a new theta using Differential Evolution.

        Returns
        -------
        theta_proposal, log_factor
        """
        n = len(population)
        if n < 2:
            raise ValueError("Population must contain at least 2 particles.")

        # sample indices of two different partner particles
        i1 = i2 = 0
        while i1 == i2:
            i1 = np.random.randint(0, n)
            i2 = np.random.randint(0, n)

        theta1 = np.asarray(population[i1], dtype=float)
        theta2 = np.asarray(population[i2], dtype=float)
        theta_arr = np.asarray(theta, dtype=float)

        # γ = γ₀ * (1 + σ_gamma * N(0,1))
        gamma = self.gamma0 * (1.0 + self.sigma_gamma * np.random.randn())

        log_factor = 0.0
        proposal = theta_arr + gamma * (theta1 - theta2)
        return proposal, log_factor

    def update(self, population: List[Any]) -> None:
        # Nothing to do; Differential Evolution does not adapt covariance here.
        return


# -------------------------------------------------------
# Stretch Move proposal
# -------------------------------------------------------


@dataclass
class StretchMove(Proposal):
    """
    Stretch move proposal (Goodman & Weare, 2010),
    i.e. the standard EMCEE-style ensemble proposal.

    Parameters
    ----------
    a : float
        Stretch parameter, usually 2.
    """

    a: float = 2.0

    def __call__(self, theta: Any, population: List[Any]) -> tuple[Any, float]:
        """
        Propose a new theta using the stretch move.

        Returns
        -------
        theta_proposal, log_factor
        """
        n = len(population)
        if n < 1:
            raise ValueError("Population must not be empty.")

        # sample index of a partner particle
        # (note: splitting in SABC ensures it's different from theta)
        i = np.random.randint(0, n)
        partner = np.asarray(population[i], dtype=float)
        theta_arr = np.asarray(theta, dtype=float)

        # z ~ g(z) ∝ 1/sqrt(z), z ∈ [1/a, a]
        # here sampled via: z = ((a - 1) * U + 1)^2 / a, U ~ Uniform(0,1)
        U = np.random.rand()
        z = ((self.a - 1.0) * U + 1.0) ** 2 / self.a

        # log factor: log(z) * (d - 1)
        d = theta_arr.size
        log_factor = np.log(z) * (d - 1)

        proposal = partner + z * (theta_arr - partner)
        return proposal, log_factor

    def update(self, population: List[Any]) -> None:
        # Nothing to adapt.
        return