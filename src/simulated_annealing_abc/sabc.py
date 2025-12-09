# simulated_annealing_abc.py

from __future__ import annotations

import math
import warnings
import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, List, Any, Sequence, Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike

from scipy.optimize import root_scalar

# If you want a progress bar; otherwise we just use range
try:
    from tqdm import trange
except ImportError:  # pragma: no cover
    trange = range  # type: ignore

# These should be implemented in the translated cdf_estimators.py and proposals.py
from .cdf_estimators import build_cdf
from .proposals import Proposal, DifferentialEvolution, update_proposal


# -------------------------------------------
# Types to hold results
# -------------------------------------------

@dataclass
class SABCState:
    """Holds state of the SABC algorithm."""

    epsilon: np.ndarray                 # current epsilon(s)
    algorithm: str                      # 'single_eps' or 'multi_eps'

    epsilon_history: List[np.ndarray]   # list of epsilon vectors over time
    rho_history: List[np.ndarray]       # history of average rho per statistic
    u_history: List[np.ndarray]         # history of average u per statistic

    cdfs_dist_prior: Callable[[ArrayLike], np.ndarray]  # G in Albert et al. (2015)

    n_simulation: int                   # total number of simulations used
    n_accept: int                       # number of accepted updates
    n_resampling: int                   # number of population resamplings
    n_population_updates: int           # number of population updates


@dataclass
class SABCResult:
    """
    Result of an SABC run.

    Attributes
    ----------
    population : list
        Vector of parameter samples from the approximate posterior.
    u : np.ndarray
        Transformed distances (n_particles x n_stats).
    rho : np.ndarray
        User-defined distances (n_particles x n_stats).
    state : SABCState
        State of the algorithm, including histories.
    """

    population: List[Any]
    u: np.ndarray
    rho: np.ndarray
    state: SABCState

    def __repr__(self) -> str:
        n_particles = len(self.population)
        mean_u = float(np.mean(self.u))
        # During initialization, n_simulation == n_particles; avoid division by zero
        denom = max(self.state.n_simulation - n_particles, 1)
        acc_rate = self.state.n_accept / denom
        eps_str = np.round(self.state.epsilon, 4)

        lines = [
            f"Approximate posterior sample with {n_particles} particles:",
            f"  - algorithm: {self.state.algorithm!r}",
            f"  - simulations used: {self.state.n_simulation}",
            f"  - number of population updates: {self.state.n_population_updates}",
            f"  - average transformed distance: {mean_u:.4g}",
            f"  - epsilon: {eps_str}",
            f"  - number of population resamplings: {self.state.n_resampling}",
            f"  - acceptance rate: {acc_rate:.4g}",
            "The sample can be accessed with the field `population`.",
            "The history of epsilon can be accessed with `state.epsilon_history`.",
            "The history of rho can be accessed with `state.rho_history`.",
            "The history of u can be accessed with `state.u_history`.",
        ]
        return "\n".join(lines)


# -------------------------------------------
# Algorithm functions
# -------------------------------------------

def update_epsilon_single_eps(u_bar: float, v: float) -> np.ndarray:
    """
    Update a single epsilon. See eq. (31) in Albert et al., Stat. Comput. 25, 2015.
    """
    if u_bar <= np.finfo(float).eps:
        eps_new = 0.0
    else:
        def f(eps: float) -> float:
            # f(eps) = eps^2 + v * eps^(3/2) - u_bar^2
            return eps**2 + v * eps**1.5 - u_bar**2

        sol = root_scalar(f, bracket=[0.0, u_bar])
        if not sol.converged:
            raise RuntimeError("Failed to find root for epsilon.")
        eps_new = float(sol.root)

    return np.array([eps_new], dtype=float)


def update_epsilon_multi_eps(u: np.ndarray, v: float) -> np.ndarray:
    """
    Update multiple epsilons. See eqs. (19–20) in Albert et al. (in preparation).

    Parameters
    ----------
    u : np.ndarray
        Transformed distances, shape (n_particles, n_stats).
    v : float
        Annealing speed parameter.

    Returns
    -------
    epsilon_new : np.ndarray
        New epsilon vector, shape (n_stats,).
    """
    if u.ndim != 2:
        raise ValueError("u must be a 2D array of shape (n_particles, n_stats).")

    n_stats = u.shape[1]
    u_bar = np.mean(u, axis=0)  # shape (n_stats,)

    # cn = (2n+2)! / ((n+1)! (n+2)!)
    # Using Python big integers via math.factorial
    cn = math.factorial(2 * n_stats + 2) / (
        math.factorial(n_stats + 1) * math.factorial(n_stats + 2)
    )
    epsilon_new = np.empty(n_stats, dtype=float)

    for i in range(n_stats):
        u_bar_i = u_bar[i]
        if u_bar_i <= np.finfo(float).eps:
            raise ZeroDivisionError(
                f"Mean u for statistic {i} is too small: {u_bar_i}"
            )

        # q = u_bar / u_bar_i
        q = u_bar / u_bar_i

        num = 1.0 + np.sum(q ** (n_stats / 2))
        den = cn * (n_stats + 1) * (u_bar_i ** (1 + n_stats / 2)) * np.prod(q)

        def g(beta: float) -> float:
            # (1 - exp(-beta) * (1 + beta)) / (beta * (1 - exp(-beta))) - u_bar_i
            num_g = 1.0 - math.exp(-beta) * (1.0 + beta)
            den_g = beta * (1.0 - math.exp(-beta))
            return num_g / den_g - u_bar_i

        # Initial guess 1 / u_bar_i as in the Julia code
        # We use a bracket around that guess to ensure root_scalar works
        x0 = 1.0 / u_bar_i
        bracket = (max(1e-12, x0 * 0.1), x0 * 10.0)
        sol = root_scalar(g, bracket=bracket)
        if not sol.converged:
            raise RuntimeError(f"Failed to find root for beta (statistic {i}).")
        beta_i = sol.root

        epsilon_new[i] = 1.0 / (beta_i + v * num / den)

    return epsilon_new


def resample_population(
    population: List[Any],
    u: np.ndarray,
    delta: float,
) -> Tuple[List[Any], np.ndarray, float]:
    """
    Resample population based on transformed distances u.

    Parameters
    ----------
    population : list
        List of parameter vectors (length n_particles).
    u : np.ndarray
        Transformed distances, shape (n_particles, n_stats).
    delta : float
        Resampling intensity.

    Returns
    -------
    population_resampled : list
    u_resampled : np.ndarray
    ess : float
        Effective sample size.
    """
    n_particles = len(population)
    u_bar = np.mean(u, axis=0)  # shape (n_stats,)

    # w = exp(-sum_i u[:, i] * delta / u_bar[i])
    # Handle broadcasting carefully
    w = np.exp(-np.sum(u * (delta / u_bar), axis=1))  # shape (n_particles,)

    # Resampling indices
    w_sum = np.sum(w)
    if w_sum == 0.0:
        raise RuntimeError("All weights are zero in resample_population.")
    p = w / w_sum
    idx_resampled = np.random.choice(n_particles, size=n_particles, replace=True, p=p)

    population_resampled = [population[i] for i in idx_resampled]
    u_resampled = u[idx_resampled, :]

    ess = w_sum**2 / np.sum(w**2)

    return population_resampled, u_resampled, ess


# -------------------------------------------
# Initialization
# -------------------------------------------

def initialization(
    f_dist: Callable[..., ArrayLike],
    prior: Any,
    *args,
    n_particles: int,
    n_simulation: int,
    v: float = 1.0,
    delta: float = 0.1,
    algorithm: str = "single_eps",
    **kwargs,
) -> SABCResult:
    """
    Initialization step for the SABC algorithm.

    Parameters
    ----------
    f_dist : callable
        Function f(theta, *args, **kwargs) -> distances (1D array-like).
    prior : object
        Prior distribution with methods .rvs() and .logpdf(theta).
    n_particles : int
        Number of particles.
    n_simulation : int
        Total number of simulations allowed.
    v : float
        Annealing speed.
    delta : float
        Resampling intensity.
    algorithm : {'single_eps', 'multi_eps'}

    Returns
    -------
    SABCResult
    """
    if n_simulation < n_particles:
        raise ValueError(
            f"`n_simulation = {n_simulation}` is too small for "
            f"{n_particles} particles."
        )

    # ---------------------
    # Initialize containers

    theta = prior.rvs()  # sample once to infer structure
    rho_init = np.asarray(f_dist(theta, *args, **kwargs), dtype=float)
    n_stats = rho_init.size

    distances_prior = np.empty((n_particles, n_stats), dtype=float)
    population: List[Any] = [None] * n_particles

    # ------------------
    # Build prior sample

    for i in range(n_particles):
        theta = prior.rvs()
        rho_i = np.asarray(f_dist(theta, *args, **kwargs), dtype=float)
        if rho_i.shape != (n_stats,):
            rho_i = rho_i.reshape(-1)
            if rho_i.size != n_stats:
                raise ValueError("Inconsistent number of statistics in f_dist.")
        population[i] = theta
        distances_prior[i, :] = rho_i

    rho_history: List[np.ndarray] = [np.mean(distances_prior, axis=0)]

    # ------------------
    # Estimate the cdf of rho under the prior

    if np.any(distances_prior < 0):
        raise ValueError("Negative distances are not allowed!")

    cdfs_dist_prior = build_cdf(distances_prior)

    u = np.empty_like(distances_prior)
    for i in range(n_particles):
        u[i, :] = cdfs_dist_prior(distances_prior[i, :])

    # ------------------
    # Resampling before setting initial epsilon

    population, u, ess = resample_population(population, u, delta)

    # Initialize epsilon
    if algorithm == "multi_eps":
        epsilon = update_epsilon_multi_eps(u, v)
    elif algorithm == "single_eps":
        epsilon = update_epsilon_single_eps(float(np.mean(u)), v)
    else:
        raise ValueError("algorithm must be 'single_eps' or 'multi_eps'.")

    u_history: List[np.ndarray] = [np.mean(u, axis=0)]
    epsilon_history: List[np.ndarray] = [epsilon.copy()]

    # We consider only n_particles draws from the prior (as in the Julia code)
    n_simulation_used = n_particles

    state = SABCState(
        epsilon=epsilon,
        algorithm=algorithm,
        epsilon_history=epsilon_history,
        rho_history=rho_history,
        u_history=u_history,
        cdfs_dist_prior=cdfs_dist_prior,
        n_simulation=n_simulation_used,
        n_accept=0,
        n_resampling=1,
        n_population_updates=0,
    )

    return SABCResult(population=population, u=u, rho=distances_prior, state=state)


# -------------------------------------------
# Population update
# -------------------------------------------

def update_population(
    population_state: SABCResult,
    f_dist: Callable[..., ArrayLike],
    prior: Any,
    *args,
    n_simulation: int,
    v: float = 1.0,
    delta: float = 0.1,
    proposal: Optional[Proposal] = None,
    resample: Optional[int] = None,
    checkpoint_history: int = 1,
    show_progressbar: bool = True,
    show_checkpoint: int = 100,
    **kwargs,
) -> SABCResult:
    """
    Update particles with `n_simulation` and apply importance sampling if needed.
    Modifies `population_state` in place and also returns it.

    Parameters
    ----------
    population_state : SABCResult
    f_dist : callable
    prior : distribution-like
    n_simulation : int
        Number of simulations to use in this update phase (total calls to f_dist).
    v : float
        Annealing speed (must be positive).
    delta : float
        Resampling intensity (must be positive).
    proposal : Proposal, optional
        Proposal object with __call__(theta, population_inactive) -> (theta_prop, log_factor).
    resample : int, optional
        After how many accepted population updates to resample.
        Default: 2 * n_particles.
    checkpoint_history : int
        Every how many population updates to store history.
    show_progressbar : bool
        If True, show a progress bar (if tqdm is available).
    show_checkpoint : int
        Every how many population updates to print a status message.

    Returns
    -------
    SABCResult
    """
    if v <= 0:
        raise ValueError("Annealing speed `v` must be positive.")
    if delta <= 0:
        raise ValueError("Resampling intensity `delta` must be positive.")

    state = population_state.state
    population = list(population_state.population)
    u = np.array(population_state.u, copy=True)
    rho = np.array(population_state.rho, copy=True)
    n_stats = u.shape[1]
    n_particles = len(population)

    epsilon = state.epsilon
    algorithm = state.algorithm
    epsilon_history = list(state.epsilon_history)
    rho_history = list(state.rho_history)
    u_history = list(state.u_history)
    n_accept = state.n_accept
    n_resampling = state.n_resampling
    cdfs_dist_prior = state.cdfs_dist_prior

    if resample is None:
        resample = 2 * n_particles

    n_population_updates = n_simulation // n_particles
    n_updates = n_population_updates * n_particles
    last_checkpoint_epsilon = 0

    if n_population_updates == 0:
        warnings.warn(
            "n_simulation is too small to perform any population update.",
            RuntimeWarning,
        )
        return population_state

    t_start = dt.datetime.now()

    # Estimate jump covariance through the proposal
    if proposal is None:
        # You may want to adapt how to infer n_params
        n_para = np.size(population[0])
        proposal = DifferentialEvolution(n_para=n_para)

    update_proposal(proposal, population)

    # Progress bar
    iterator = trange(
        1,
        n_population_updates + 1,
        desc=f"{n_population_updates} population updates:",
        disable=not show_progressbar,
    )

    for ix in iterator:
        # Split population indices in two halves
        mid = n_particles // 2
        batch_1 = range(0, mid)
        batch_2 = range(mid, n_particles)

        n_accept_tmp = 0

        for active, inactive in ((batch_1, batch_2), (batch_2, batch_1)):
            population_inactive = [population[j] for j in inactive]

            for i in active:
                theta_current = population[i]

                theta_proposal, log_factor = proposal(theta_current, population_inactive)

                lp_proposal = prior.logpdf(theta_proposal)
                if not np.isfinite(lp_proposal):
                    log_accept_prob = -np.inf
                else:
                    lp_current = prior.logpdf(theta_current)
                    rho_proposal = np.asarray(
                        f_dist(theta_proposal, *args, **kwargs), dtype=float
                    )
                    if rho_proposal.shape != (n_stats,):
                        rho_proposal = rho_proposal.reshape(-1)
                        if rho_proposal.size != n_stats:
                            raise ValueError(
                                "Inconsistent number of statistics in f_dist."
                            )
                    u_proposal = cdfs_dist_prior(rho_proposal)

                    log_accept_prob = (
                        lp_proposal
                        - lp_current
                        + np.sum((u[i, :] - u_proposal) / epsilon)
                        + log_factor
                    )

                if math.log(np.random.rand()) < log_accept_prob:
                    population[i] = theta_proposal
                    u[i, :] = u_proposal
                    rho[i, :] = rho_proposal
                    n_accept_tmp += 1

        n_accept += n_accept_tmp

        # Resampling
        if n_accept >= (n_resampling + 1) * resample:
            population, u, ess = resample_population(population, u, delta)
            n_resampling += 1

        # Update epsilon and proposal distribution
        update_proposal(proposal, population)

        if algorithm == "multi_eps":
            epsilon = update_epsilon_multi_eps(u, v)
        elif algorithm == "single_eps":
            epsilon = update_epsilon_single_eps(float(np.mean(u)), v)
        else:
            raise ValueError("algorithm must be 'single_eps' or 'multi_eps'.")

        # Progress / history
        if show_checkpoint and ix % show_checkpoint == 0:
            elapsed = dt.datetime.now() - t_start
            eta = elapsed / ix * (n_population_updates - ix)
            if eta.total_seconds() > 1:
                eta_str = str(eta).split(".")[0]
            else:
                eta_str = "< 1 second"
            print(
                f"Update {ix} / {n_population_updates}. "
                f"Average transformed distance: {np.mean(u):.4g}, "
                f"epsilon: {np.round(epsilon, 4)}, ETA: {eta_str}",
                flush=True,
            )

        if ix % checkpoint_history == 0:
            epsilon_history.append(epsilon.copy())
            u_history.append(np.mean(u, axis=0))
            rho_history.append(np.mean(rho, axis=0))
            last_checkpoint_epsilon = ix

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                eps=np.round(epsilon, 4), avg_u=float(np.round(np.mean(u), 4))
            )

    # Store the last epsilon value if not already done
    if last_checkpoint_epsilon != n_population_updates:
        epsilon_history.append(epsilon.copy())
        u_history.append(np.mean(u, axis=0))
        rho_history.append(np.mean(rho, axis=0))

    # Update state
    state.epsilon = epsilon
    state.epsilon_history = epsilon_history
    state.u_history = u_history
    state.rho_history = rho_history
    state.n_simulation += n_updates
    state.n_accept = n_accept
    state.n_resampling = n_resampling
    state.n_population_updates += n_population_updates

    population_state.population = population
    population_state.u = u
    population_state.rho = rho

    print(
        f"All particles have been updated {n_population_updates} times.",
        flush=True,
    )
    return population_state


# -------------------------------------------
# Main SABC wrapper
# -------------------------------------------

def sabc(
    f_dist: Callable[..., ArrayLike],
    prior: Any,
    *args,
    n_particles: int = 100,
    n_simulation: int = 10_000,
    algorithm: str = "single_eps",
    proposal: Optional[Proposal] = None,
    resample: Optional[int] = None,
    v: float = 1.0,
    delta: float = 0.1,
    checkpoint_history: int = 1,
    show_progressbar: bool = True,
    show_checkpoint: int = 100,
    **kwargs,
) -> SABCResult:
    """
    Simulated Annealing Approximate Bayesian Computation (SABC).

    Parameters
    ----------
    f_dist : callable
        Function that returns one or more distances between the observation
        and a random sample from the likelihood. Signature:
        f_dist(theta, *args, **kwargs) -> 1D array-like of distances.
    prior : distribution-like
        Object with methods .rvs() and .logpdf(theta) defining the prior.
    n_particles : int
        Desired number of particles.
    n_simulation : int
        Maximum number of simulations from f_dist.
    algorithm : {'single_eps', 'multi_eps'}
        Algorithm for tolerance.
    proposal : Proposal, optional
        Proposal distribution object. If None, a DifferentialEvolution
        proposal is constructed.
    resample : int, optional
        After how many accepted population updates to resample.
        Default: 2 * n_particles.
    v : float
        Tuning parameter for annealing speed (must be positive).
    delta : float
        Tuning parameter for resampling intensity (must be positive and small).
    checkpoint_history : int
        Every how many population updates distances and epsilons are stored.
    show_progressbar : bool
        If True, show progress bar (if tqdm is available).
    show_checkpoint : int
        Every how many population updates the algorithm state is displayed.
    kwargs :
        Further keyword arguments passed to f_dist.

    Returns
    -------
    SABCResult
    """
    if algorithm not in ("multi_eps", "single_eps"):
        raise ValueError(
            f"Argument `algorithm` must be 'multi_eps' or 'single_eps', not {algorithm!r}."
        )

    # ------------------------------------
    # Initialization

    population_state = initialization(
        f_dist,
        prior,
        *args,
        n_particles=n_particles,
        n_simulation=n_simulation,
        v=v,
        delta=delta,
        algorithm=algorithm,
        **kwargs,
    )

    # ------------------------------
    # Sampling

    n_sim_remaining = n_simulation - population_state.state.n_simulation
    if n_sim_remaining < n_particles:
        warnings.warn(
            "`n_simulation` too small to update all particles at least once.",
            RuntimeWarning,
        )

    population_state = update_population(
        population_state,
        f_dist,
        prior,
        *args,
        n_simulation=n_sim_remaining,
        resample=resample if resample is not None else 2 * n_particles,
        proposal=proposal,
        v=v,
        delta=delta,
        checkpoint_history=checkpoint_history,
        show_progressbar=show_progressbar,
        show_checkpoint=show_checkpoint,
        **kwargs,
    )

    return population_state