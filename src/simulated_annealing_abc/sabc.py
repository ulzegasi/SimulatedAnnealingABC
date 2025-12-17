# simulated_annealing_abc.py (Python 3.14)

import math
import warnings
import datetime as dt
import sys
from dataclasses import dataclass
from typing import TextIO

import numpy as np
from scipy.optimize import root_scalar

try:
    from tqdm import trange
    _use_tqdm = True
except ImportError:
    trange = range
    _use_tqdm = False

from simulated_annealing_abc.cdf_estimators import build_cdf
from simulated_annealing_abc.proposals import Proposal, DifferentialEvolution, update_proposal


# -------------------------------------------
# Result containers
# -------------------------------------------

@dataclass
class SABCState:
    epsilon: np.ndarray # epsilon = temperature
    algorithm: str      # the algorithm used
    
    # Containers to store trajectories
    epsilon_history: list
    rho_history: list
    u_history: list

    cdfs_dist_prior: object  # callable from build_cdf
    # function G in Albert et al., Statistics and Computing 25, 2015

    n_simulation: int         # number of simulations
    n_accept: int             # number of accepted updates
    n_resampling: int         # number of population resamplings
    n_population_updates: int # number of population updates


@dataclass
class SABCResult:
    population: list
    u: np.ndarray   # transformed distances
    rho: np.ndarray # user-defined distances
    state: SABCState


# -------------------------------------------
# Epsilon updates
# -------------------------------------------

# Update a single epsilon. 
# See eq(31) in Albert et al., Statistics and Computing 25, 2015
def update_epsilon_single_eps(u_bar: float, v: float) -> np.ndarray:
    if u_bar <= np.finfo(float).eps:
        return np.array([0.0], dtype=float)

    f = lambda eps: eps**2 + v*eps**1.5 - u_bar**2

    sol = root_scalar(f, bracket=(0.0, u_bar))
    if not sol.converged:
        raise RuntimeError("Failed to find root for epsilon.")
    return np.array([float(sol.root)], dtype=float)

# Update multiple epsilons.
def update_epsilon_multi_eps(u: np.ndarray, v: float) -> np.ndarray:
    if u.ndim != 2:
        raise ValueError("u must be 2D (n_particles, n_stats).")

    n_stats = u.shape[1]
    u_bar = np.mean(u, axis=0) # mean over particles (vector of size n_stats)

    cn = math.factorial(2 * n_stats + 2) / (
        math.factorial(n_stats + 1) * math.factorial(n_stats + 2)
    )

    epsilon_new = np.empty(n_stats, dtype=float)

    for i in range(n_stats):
        u_bar_i = u_bar[i]
        if u_bar_i <= np.finfo(float).eps:
            raise ZeroDivisionError(f"Mean u for statistic {i} too small.")

        q = u_bar / u_bar_i
        num = 1.0 + np.sum(q ** (n_stats / 2))
        den = cn * (n_stats + 1) * (u_bar_i ** (1 + n_stats / 2)) * np.prod(q)

        def g(beta: float) -> float:
            num_g = 1.0 - math.exp(-beta) * (1.0 + beta)
            den_g = beta * (1.0 - math.exp(-beta))
            return num_g / den_g - u_bar_i

        x0 = 1.0 / u_bar_i
        bracket = (max(1e-12, x0 * 0.1), x0 * 10.0)
        sol = root_scalar(g, bracket=bracket)
        if not sol.converged:
            raise RuntimeError(f"Failed to find root for beta (stat {i}).")

        beta_i = float(sol.root)
        epsilon_new[i] = 1.0 / (beta_i + v * num / den)

    return epsilon_new


# -------------------------------------------
# Resampling
# -------------------------------------------

def resample_population(population: list, u: np.ndarray, delta: float):
    n_particles = len(population)
    u_bar = np.mean(u, axis=0) # mean over particles (vector of size n_stats)

    w = np.exp(-np.sum(u * (delta / u_bar), axis=1))
    w_sum = np.sum(w)
    if w_sum == 0.0:
        raise RuntimeError("All weights are zero in resample_population.")

    p = w / w_sum
    idx = np.random.choice(n_particles, size=n_particles, replace=True, p=p)

    population_resampled = [population[i] for i in idx]
    u_resampled = u[idx, :]

    ess = w_sum**2 / np.sum(w**2)
    return population_resampled, u_resampled, ess


# -------------------------------------------
# Check if prior is valid
# -------------------------------------------

def _check_prior(prior):
    for name in ("rvs", "logpdf"):
        if not hasattr(prior, name):
            raise TypeError(
                "prior must provide methods .rvs() and .logpdf(theta). "
                f"Missing: {name}"
            )

# -------------------------------------------
# Detect interactive terminal vs. redirected/logged output
# -------------------------------------------
            
def is_logging(stream: TextIO) -> bool:
    """True if stream is not an interactive terminal (i.e., redirected/logged)."""
    return not stream.isatty()

# -------------------------------------------
# Print messages to stderr
# -------------------------------------------

def info(msg):
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)

# -------------------------------------------
# Initialization
# -------------------------------------------

def initialization(
    f_dist,
    prior,
    *args,
    n_particles: int,
    n_simulation: int,
    v: float = 1.0,
    delta: float = 0.1,
    algorithm: str = "single_eps",
    **kwargs,
) -> SABCResult:
    
    _check_prior(prior)
    
    if n_simulation < n_particles:
        raise ValueError(f"`n_simulation={n_simulation}` too small for {n_particles} particles.")

    info(f"Initialization for '{algorithm}'")
    # ---------------------
    # Initialize containers
    
    theta = prior.rvs()  # take a random sample
    # generate dummy distances to initialize containers
    rho_init = np.asarray(f_dist(theta, *args, **kwargs), dtype=float)
    if rho_init.ndim != 1:
        raise ValueError("f_dist must return a 1D array of distances.")
    n_stats = rho_init.size
    rho = np.empty((n_particles, n_stats), dtype=float)
    population = [None] * n_particles
    
    # ------------------
    # Build prior sample ----> CAN BE PARALLELIZED <----

    for i in range(n_particles):
        theta = prior.rvs()
        rho_i = np.asarray(f_dist(theta, *args, **kwargs), dtype=float)
        population[i] = theta
        rho[i, :] = rho_i

    if np.any(rho < 0):
        raise ValueError("Negative distances are not allowed!")

    rho_history = [np.mean(rho, axis=0)]
    
    # ------------------
    # Estimate the cdf of ρ given the prior

    cdfs_dist_prior = build_cdf(rho)
    # Transformed distances
    u = np.empty_like(rho)
    for i in range(n_particles):
        u[i, :] = cdfs_dist_prior(rho[i, :])
    
    # ------------------
    # Resampling before setting initial epsilon

    population, u, ess = resample_population(population, u, delta)

    if algorithm == "multi_eps":
        epsilon = update_epsilon_multi_eps(u, v)
    elif algorithm == "single_eps":
        epsilon = update_epsilon_single_eps(np.mean(u), v)
    else:
        raise ValueError("algorithm must be 'single_eps' or 'multi_eps'.")

    # ---------------------
    # Collect parameters and state of the algorithm
    
    state = SABCState(
        epsilon=epsilon,
        algorithm=algorithm,
        epsilon_history=[epsilon.copy()],
        rho_history=rho_history,
        u_history=[np.mean(u, axis=0)],
        cdfs_dist_prior=cdfs_dist_prior,
        n_simulation=n_particles,
        # N.B.: we consider only n_particles draws from the prior
        # and neglect the first call to f_dist ('initialization of containers')
        n_accept=0,
        n_resampling=1,
        n_population_updates=0,
    ) 
    return SABCResult(population=population, u=u, rho=rho, state=state)


# -------------------------------------------
# Population update
# -------------------------------------------

def update_population(
    population_state: SABCResult,
    f_dist,
    prior,
    *args,
    n_simulation: int,
    v: float = 1.0,
    delta: float = 0.1,
    proposal: Proposal | None = None,
    resample: int | None = None,
    checkpoint_history: int = 1,
    show_progressbar: bool | None = None,
    show_checkpoint: float | int | None = None,
    **kwargs,
) -> SABCResult:
    
    if v <= 0:
        raise ValueError("Annealing speed v must be positive.")
    if delta <= 0:
        raise ValueError("Resampling parameter delta must be positive.")
    
    _check_prior(prior)

    logging_stderr = is_logging(sys.stderr)
    if show_progressbar is None:
        show_progressbar = not logging_stderr
        
    if show_checkpoint is None:
        show_checkpoint = 100 if logging_stderr else float("inf")

    # ---------------------
    # Copy arrays that will be modified locally (population, u, rho), 
    # but keep a single evolving 'state' object that tracks histories/counters.
    
    state = population_state.state
    population = list(population_state.population)
    u = np.array(population_state.u, copy=True)
    rho = np.array(population_state.rho, copy=True)
    n_particles = len(population)
    n_stats = u.shape[1]
    
    # ---------------------
    # Set up proposal mechanism and 
    # resampling interval, if not provided
    
    if proposal is None:
        n_para = np.size(population[0])
        proposal = DifferentialEvolution(n_para=n_para)
    if resample is None:
        resample = 2 * n_particles
        
    # ---------------------
    # Estimate jump covariance from current population
    
    update_proposal(proposal, population)

    # ---------------------
    # Each population update requires n_particles simulations
    
    n_population_updates = n_simulation // n_particles
    if n_population_updates == 0:
        warnings.warn("n_simulation too small to perform any population update.", RuntimeWarning)
        return population_state
    n_updates = n_population_updates * n_particles   # total particle updates / f_dist calls
    last_checkpoint_epsilon = 0.0                    # or 0, depending on your epsilon type

    # ---------------------
    # To estimate ETA
    
    t_start = dt.datetime.now()
    
    # ---------------------
    # Iterator for outer loop (1 -> n_population_updates)
    
    if _use_tqdm:
        iterator = trange(
            1, n_population_updates + 1,
            desc=f"{n_population_updates} population updates:",
            disable=not show_progressbar,
            file=sys.stderr,
        )
    else:
        iterator = range(1, n_population_updates + 1)

    # ---------------------
    # Loop over population updates
    # At each iteration ix all particles are updated
    
    for ix in iterator:
        # Split population indices in two halves
        mid = n_particles // 2
        batch_1 = range(0, mid)
        batch_2 = range(mid, n_particles)

        n_accept_tmp = 0

        for active, inactive in ((batch_1, batch_2), (batch_2, batch_1)):
            
            pop_inactive = [population[j] for j in inactive]

            # ----> INNER LOOP CAN BE PARALLELIZED <----
            for i in active:
                # Generate proposal
                theta_cur = population[i]
                theta_prop, log_factor = proposal(theta_cur, pop_inactive)
                # Evaluate log prior at proposed theta
                lprior_prop = prior.logpdf(theta_prop)
                # Compute acceptance probability    
                if not np.isfinite(lprior_prop):
                    log_accept_prob = -np.inf
                else:
                    lprior_cur = prior.logpdf(theta_cur)
                    rho_prop = np.asarray(f_dist(theta_prop, *args, **kwargs), dtype=float)
                    if rho_prop.size != n_stats:
                        raise ValueError("Inconsistent number of statistics in f_dist.")
                    u_prop = state.cdfs_dist_prior(rho_prop)

                    log_accept_prob = (
                        lprior_prop - lprior_cur
                        + np.sum((u[i, :] - u_prop) / state.epsilon)
                        + log_factor
                    )

                if math.log(np.random.rand()) < log_accept_prob:
                    population[i] = theta_prop
                    u[i, :] = u_prop
                    rho[i, :] = rho_prop
                    n_accept_tmp += 1
            # END of inner loop over active particles
            
        state.n_accept += n_accept_tmp # careful here, avoid race conditions when parallelizing

        # ---------------------
        # Resample population if needed
        
        if state.n_accept >= (state.n_resampling + 1) * resample:
            population, u, ess = resample_population(population, u, delta)
            state.n_resampling += 1

        # ---------------------
        # Update proposal distribution and epsilon
        
        update_proposal(proposal, population)

        if state.algorithm == "multi_eps":
            state.epsilon = update_epsilon_multi_eps(u, v)
        else:
            state.epsilon = update_epsilon_single_eps(float(np.mean(u)), v)

        # ---------------------
        # Update population statistics and print progress
        
        state.n_population_updates += 1
        state.n_simulation += n_particles

        if show_checkpoint and ix % show_checkpoint == 0:
            elapsed = dt.datetime.now() - t_start
            eta = elapsed / ix * (n_population_updates - ix)
            eta_str = str(eta).split(".")[0] if eta.total_seconds() > 1 else "< 1 second"
            print(
                f"Update {ix}/{n_population_updates}  "
                f"avg_u={np.mean(u):.4g}  eps={np.round(state.epsilon, 4)}  ETA={eta_str}",
                flush=True,
            )
        # ---------------------
        # Store histories
        
        if ix % checkpoint_history == 0:
            state.epsilon_history.append(state.epsilon.copy())
            state.u_history.append(np.mean(u, axis=0))
            state.rho_history.append(np.mean(rho, axis=0))

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                ε=f"{state.epsilon:.4g}",
                avg_dist=f"{np.mean(u):.4g}",
        )
    # END of main loop over population updates
    
    population_state.u = u
    population_state.rho = rho
    
    info(f"All particles have been updated {n_population_updates} times.")

    return population_state


# -------------------------------------------
# Main
# -------------------------------------------

def sabc(
    f_dist,
    prior,
    *args,
    n_particles: int = 1000,
    n_simulation: int = 10_000,
    algorithm: str = "single_eps",
    proposal: Proposal | None = None,
    resample: int | None = None,
    v: float = 1.0,
    delta: float = 0.1,
    checkpoint_history: int = 1,
    show_progressbar: bool | None = None,
    show_checkpoint: float | int | None = None,
    **kwargs,
) -> SABCResult:
    
    if algorithm not in ("single_eps", "multi_eps"):
        raise ValueError("algorithm must be 'single_eps' or 'multi_eps'.")

    # ---------------------
    # Initialization
        
    pop_state = initialization(
        f_dist, prior, *args,
        n_particles=n_particles,
        n_simulation=n_simulation,
        v=v, delta=delta,
        algorithm=algorithm,
        **kwargs,
    )
    
    # ---------------------
    # Sampling / Population updates

    n_sim_remaining = n_simulation - pop_state.state.n_simulation
    if n_sim_remaining < n_particles:
        warnings.warn("`n_simulation` too small to update all particles at least once.", RuntimeWarning)

    return update_population(
        pop_state, f_dist, prior, *args,
        n_simulation=n_sim_remaining,
        v=v, delta=delta,
        proposal=proposal,
        resample=resample,
        checkpoint_history=checkpoint_history,
        show_progressbar=show_progressbar,
        show_checkpoint=show_checkpoint,
        **kwargs,
    )
