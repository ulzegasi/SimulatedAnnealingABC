# proposals.py  (Python 3.14)

from dataclasses import dataclass
import numpy as np


# -------------------------------------------------------
# Base proposal type
# -------------------------------------------------------

class Proposal:
    """Base class for proposal generators."""
    def update(self, population):
        """Update internal state (e.g. covariance)."""
        return

# Interface to update the proposal state
def update_proposal(proposal: Proposal, population):
    """Update the proposal's internal state."""
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

    def __call__(self, theta, population):
        log_factor = 0.0

        # 1D case: Sigma is a scalar variance
        if np.isscalar(self.Sigma):
            var = float(self.Sigma)
            if var <= 0.0:
                raise RuntimeError("RandomWalk Sigma not updated yet.")
            step = np.random.normal(loc=0.0, scale=np.sqrt(var))
            return theta + step, log_factor

        # nD case: Sigma is a covariance matrix
        cov = np.asarray(self.Sigma, dtype=float)
        if cov.ndim != 2:
            raise ValueError("RandomWalk Sigma must be a covariance matrix.")
        d = cov.shape[0]
        step = np.random.multivariate_normal(mean=np.zeros(d), cov=cov)
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
        return theta_arr + step, log_factor

    def update(self, population):
        # 1D case
        if np.isscalar(self.Sigma):
            arr = np.array([np.atleast_1d(p)[0] for p in population], dtype=float)
            cov = np.cov(arr, bias=False)  # variance in 1D
            self.Sigma = max(self.beta * float(cov), 1e-12)
            return

        # nD case
        arr = np.stack([np.asarray(p, dtype=float) for p in population], axis=0)
        cov = np.cov(arr, rowvar=False, bias=False)
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

    def __call__(self, theta, population):
        n = len(population)
        if n < 2:
            raise ValueError("Population must contain at least 2 particles.")

        # pick two distinct partners
        i1, i2 = np.random.choice(n, size=2, replace=False)

        theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
        theta1 = np.atleast_1d(np.asarray(population[i1], dtype=float))
        theta2 = np.atleast_1d(np.asarray(population[i2], dtype=float))

        gamma = self.gamma0 * (1.0 + self.sigma_gamma * np.random.randn())

        log_factor = 0.0
        proposal = theta_arr + gamma * (theta1 - theta2)
        if proposal.size == 1:
            return float(proposal[0]), log_factor
        return proposal, log_factor

    def update(self, population):
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
    
    def __call__(self, theta, population):
        n = len(population)
        if n < 1:
            raise ValueError("Population must not be empty.")

        i = np.random.randint(0, n)
        
        # Enforce 1D shape even for scalar parameters
        partner = np.atleast_1d(np.asarray(population[i], dtype=float))
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))

        U = np.random.rand()
        z = ((self.a - 1.0) * U + 1.0) ** 2 / self.a

        d = theta_arr.size
        log_factor = np.log(z) * (d - 1)

        proposal = partner + z * (theta_arr - partner)
        if proposal.size == 1:
            return float(proposal[0]), log_factor
        
        return proposal, log_factor

    def update(self, population):
        return
