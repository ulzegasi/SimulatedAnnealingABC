# sabc.py — core Simulated Annealing ABC algorithm
# Python 3.14

import math
import warnings
import datetime as dt
import sys
from dataclasses import dataclass
from typing import TextIO, Callable

import numpy as np
from scipy.optimize import root_scalar

# ------------------------------------------------------------------
# Environment detection (terminal / notebook / batch)
# ------------------------------------------------------------------

def running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False

def is_interactive() -> bool:
    # Jupyter notebook => interactive
    if running_in_notebook():
        return True
    # Real terminal => interactive
    return sys.stderr.isatty() or sys.stdout.isatty()

# ------------------------------------------------------------------
# Try to enable tqdm progress bars; fall back to a plain range iterator
# if tqdm is not available.
# NOTE 1: tqdm is a Python library that displays progress bars for loops.
# NOTE 2: The name comes from the Arabic taqaddum (تقدّم), meaning “progress”
# ------------------------------------------------------------------

try:
    from tqdm.auto import trange
    _use_tqdm = True
except ImportError:
    trange = range
    _use_tqdm = False

from simulated_annealing_abc.cdf_estimators import build_cdf
from simulated_annealing_abc.proposals import (
    Proposal,
    DifferentialEvolution,
    RandomWalk,
    StretchMove,
    update_proposal,
)

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

    # Fixed "prior-distance" reference used to build the CDF mapping
    rho_prior: np.ndarray  # prior distances computed at initialization from prior samples
    cdfs_dist_prior: Callable[[np.ndarray], np.ndarray] | None # callable function or None
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
    if u_bar <= 1e-12:
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
    u_bar = np.mean(u, axis=0)  # mean over particles (vector of size n_stats)
    u_bar = np.maximum(u_bar, 1e-12)

    cn = math.factorial(2 * n_stats + 2) / (
        math.factorial(n_stats + 1) * math.factorial(n_stats + 2)
    )

    epsilon_new = np.empty(n_stats, dtype=float)

    for i in range(n_stats):
        u_bar_i = float(u_bar[i])

        # A positive beta root exists only for u_bar_i < 0.5
        # (as beta -> 0, ratio -> 1/2; as beta -> inf, ratio -> 0).
        if u_bar_i >= 0.5:
            u_bar_i = 0.5 - 1e-12

        q = u_bar / u_bar_i
        num = 1.0 + np.sum(q ** (n_stats / 2))
        den = cn * (n_stats + 1) * (u_bar_i ** (1 + n_stats / 2)) * np.prod(q)

        def g(beta: float) -> float:
            # For very small beta, use a series expansion to avoid 0/0 cancellation:
            # ratio = 1/2 - beta/12 + O(beta^2)
            if beta < 1e-6:
                return (0.5 - beta / 12.0) - u_bar_i

            # Otherwise, stable computation using expm1
            em1 = math.expm1(-beta)          # e^{-beta} - 1
            exp_neg = em1 + 1.0              # e^{-beta}

            num_g = 1.0 - exp_neg * (1.0 + beta)
            den_g = beta * (1.0 - exp_neg)

            return (num_g / den_g) - u_bar_i

        # --- robust bracketing ---
        a = 1e-6
        b = max(1.0, 10.0 / u_bar_i)

        fa = g(a)
        fb = g(b)

        # Expand upper bound until sign change (should happen quickly)
        k = 0
        while fa * fb > 0 and k < 60:
            b *= 2.0
            fb = g(b)
            k += 1

        if fa * fb > 0:
            raise RuntimeError(
                f"Failed to bracket beta root for stat {i}: "
                f"u_bar_i={u_bar_i:.6g}, g(a)={fa:.6g}, g(b)={fb:.6g}"
            )

        sol = root_scalar(g, bracket=(a, b), method="brentq")
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
    u_bar = np.maximum(u_bar, 1e-12)   # avoid division by ~0

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
    
    rho_prior = rho.copy()
    cdfs_dist_prior = build_cdf(rho_prior)
    # Transformed distances
    u = np.empty_like(rho_prior)
    for i in range(n_particles):
        u[i, :] = cdfs_dist_prior(rho_prior[i, :])
    
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
        rho_prior=rho_prior,
        cdfs_dist_prior=cdfs_dist_prior,
        n_simulation=n_particles,
        # N.B.: we consider only n_particles draws from the prior
        # and neglect the first call to f_dist ('initialization of containers')
        n_accept=0,
        n_resampling=1,
        n_population_updates=0,
    ) 
    return SABCResult(population=population, u=u, rho=rho_prior, state=state)


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

    # ---------------------
    # Decide interactive vs logging behavior

    interactive = is_interactive()

    if show_progressbar is None:
        show_progressbar = interactive

    if show_checkpoint is None:
        show_checkpoint = None if interactive else 100
    # ---------------------
    # Extract variables from population_state for easier access
    # Updates inside the loop will mutate the original objects in population_state

    state = population_state.state
    population = population_state.population
    u = population_state.u
    rho = population_state.rho
    # population, u, rho and 
    # population_state.population, population_state.u, population_state.rho
    # refer to the same objects in memory, respectively. 
    # Any in-place modification (e.g., population[i] = ...) affects both. Similarly for u and rho.
    # BE CAREFUL: Resampling will rebind the local names 'population' and 'u' to new objects 
    # We will need to reattach them (SEE BELOW)
    n_particles = len(population)
    n_stats = u.shape[1]
    
    # ---------------------
    # Rebuild CDF mapping 
    # if we are resuming from a serialized checkpoint
    if state.cdfs_dist_prior is None:
        state.cdfs_dist_prior = build_cdf(state.rho_prior)
    
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
            # Local names 'population' and 'u' now refer to new objects created by resampling
            # We must explicitly REATTACH the new objects to the population_state
            population_state.population = population
            population_state.u = u
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

        if (not show_progressbar) and (show_checkpoint is not None) and ix % show_checkpoint == 0:
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

        # ---------------------
        # Update progress bar if required
        
        if hasattr(iterator, "set_postfix"):
            if np.size(state.epsilon) == 1:
                iterator.set_postfix(
                    eps=f"{float(state.epsilon):.4g}",
                    avg_dist=f"{np.mean(u):.4g}",
                )
            else:
                iterator.set_postfix(
                    avg_eps=f"{float(np.mean(state.epsilon)):.4g}",
                    avg_dist=f"{np.mean(u):.4g}",
                )
    # END of main loop over population updates

    # In principle there is NO NEED to reassign population, u and rho to the population_state
    # They are already the same objects.
    # The following lines are therefore redundant (but cheap)
    population_state.population = population
    population_state.u = u
    population_state.rho = rho
    
    # Make result pickle-safe / restart-friendly (closures don't serialize reliably)
    state.cdfs_dist_prior = None
    
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
