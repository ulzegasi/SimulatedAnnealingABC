# proposals.py  (Python 3.14)

from dataclasses import dataclass
import numpy as np


# -------------------------------------------------------
# Base proposal type
# -------------------------------------------------------
class Proposal:
    """Base class for proposal generators."""
    def update(self, population: np.ndarray) -> None:
        return


def update_proposal(proposal: Proposal, population: np.ndarray) -> None:
    proposal.update(population)


# -------------------------------------------------------
# Random Walk proposal
# -------------------------------------------------------
@dataclass(init=False)
class RandomWalk(Proposal):
    """
    Gaussian random walk proposal.

    Parameters
    ----------
    beta : float
        Mixing parameter in (0, 1].
    Sigma : float or np.ndarray
        Jump variance (1D) or covariance matrix (nD), adapted from population.
    """
    beta: float
    Sigma: float | np.ndarray  # scalar variance or covariance matrix

    def __init__(self, *, beta=0.8, n_para=1):
        if not (0.0 < beta <= 1.0):
            raise ValueError("Mixing parameter `beta` must be between 0 and 1.")
        self.beta = float(beta)
        if n_para == 1:
            self.Sigma = -1.0
        else:
            self.Sigma = -np.ones((n_para, n_para), dtype=float)

    def __call__(self, theta: np.ndarray, population: np.ndarray):
        log_factor = 0.0

        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 1:
            raise ValueError("theta must be a 1D array.")
        
        # 1D case: Sigma is a scalar variance
        if np.isscalar(self.Sigma):
            var = float(self.Sigma)
            if var <= 0.0:
                raise RuntimeError("RandomWalk Sigma not updated yet.")
            step = np.random.normal(loc=0.0, scale=np.sqrt(var))
            return np.array([theta[0] + step], dtype=float), log_factor

        # nD case: Sigma is a covariance matrix
        cov = np.asarray(self.Sigma, dtype=float)
        if cov.ndim != 2:
            raise ValueError("RandomWalk Sigma must be a covariance matrix.")
        d = cov.shape[0]
        if theta.size != d:
            raise ValueError("theta dimension does not match Sigma.")
        step = np.random.multivariate_normal(mean=np.zeros(d), cov=cov)
        return theta + step, log_factor

    def update(self, population: np.ndarray)  -> None:
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
        d = cov.shape[0]
        self.Sigma = self.beta * (cov + 1e-8 * np.eye(d))


# -------------------------------------------------------
# Differential Evolution proposal
# -------------------------------------------------------
@dataclass(init=False)
class DifferentialEvolution(Proposal):
    """
    Differential Evolution proposal.

    If `n_para` is given, uses gamma0 = 2.38 / sqrt(2*n_para),
    matching the typical DE scaling used in ensemble samplers.
    """
    gamma0: float
    sigma_gamma: float

    def __init__(self, *, gamma0=None, n_para=None, sigma_gamma=1e-5):
        if (gamma0 is None) == (n_para is None):  # Error if both are None or both are provided
            raise ValueError("Provide exactly one of `gamma0` or `n_para`.")
        if gamma0 is None:
            gamma0 = 2.38 / np.sqrt(2.0 * float(n_para))
        self.gamma0 = float(gamma0)
        self.sigma_gamma = float(sigma_gamma)

    def __call__(self, theta: np.ndarray, population: np.ndarray):
        """
        theta: (d,)
        population: (m, d)  (this can be the inactive half view)
        """
        pop = population
        if pop.ndim != 2:
            raise ValueError("population must be 2D (m, d).")
        m, d = pop.shape
        if m < 2:
            raise ValueError("Population must contain at least 2 particles.")
        if theta.ndim != 1 or theta.size != d:
            raise ValueError("theta must be a 1D array of length d.")
        # pick two distinct partners
        i1, i2 = np.random.choice(m, size=2, replace=False)
        theta1 = pop[i1, :]
        theta2 = pop[i2, :]

        gamma = self.gamma0 * (1.0 + self.sigma_gamma * np.random.randn())
        log_factor = 0.0
        proposal = theta + gamma * (theta1 - theta2)
        return proposal, log_factor

    def update(self, population: np.ndarray) -> None:
        return


# -------------------------------------------------------
# Stretch Move proposal
# -------------------------------------------------------
@dataclass
class StretchMove(Proposal):
    """
    Stretch move proposal (Goodman & Weare / emcee-style).
    
    Parameter
    ---------
    a : float
        Stretch scale parameter. Must be > 1.
        Common default is a=2.0.
    """
    a: float = 2.0

    def __post_init__(self):
        # Validate parameter
        if self.a <= 1.0:
            raise ValueError("StretchMove parameter 'a' must be > 1.")
    
    def __call__(self, theta: np.ndarray, population: np.ndarray):
        pop = population
        if pop.ndim != 2:
            raise ValueError("population must be 2D (m, d).")
        m, d = pop.shape
        if m < 1:
            raise ValueError("Population must not be empty.")
        
        if theta.ndim != 1 or theta.size != d:
            raise ValueError("theta must be a 1D array of length d.")

        i = np.random.randint(0, m)
        partner = pop[i, :]

        U = np.random.rand()
        z = ((self.a - 1.0) * U + 1.0) ** 2 / self.a

        log_factor = np.log(z) * (d - 1)
        proposal = partner + z * (theta - partner)
        return proposal, log_factor

    def update(self, population: np.ndarray) -> None:
        return
