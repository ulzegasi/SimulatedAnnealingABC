"""sabc.py — core Simulated Annealing ABC algorithm."""

import logging
import math
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import root_scalar

from .cdf_estimators import CDF1D, CDFMulti, build_cdf
from .helper import track_progress
from .proposals import (
    DifferentialEvolution,
    Proposal,
    update_proposal,
)

LOG = logging.getLogger(__name__)


# -------------------------------------------
# Configuration
# -------------------------------------------
@dataclass
class SABCConfig:
    """Full configuration for an SABC run.

    Args:
        f_dist: Distance function. Either from ``make_f_dist()`` or hand-written.
            Signature: ``f_dist(theta) -> np.ndarray`` or ``f_dist(theta, out=buf) -> np.ndarray``.
        prior: Prior distribution object with ``.rvs(rng)`` and ``.logpdf(theta)`` methods.
        n_particles: Number of particles in the population.
        v: Annealing speed parameter (must be positive).
        delta: Resampling parameter (must be positive).
        algorithm: Epsilon update strategy, ``"single_eps"`` or ``"multi_eps"``.
        resample: Resampling interval in accepted proposals.
            Defaults to ``2 * n_particles`` if ``None``.
        proposal: Proposal mechanism (e.g. ``DifferentialEvolution``, ``RandomWalk``,
            ``StretchMove``). Defaults to ``DifferentialEvolution`` if ``None``.
        rng: NumPy random number generator for algorithm randomness.
        seed: Seed for creating an RNG (alternative to ``rng``).
        checkpoint_history: Record histories every N population updates.
        show_progressbar: Show a progress bar (tqdm/rich) if available.
        show_checkpoint: Log progress every N population updates.
    """

    # Problem definition
    f_dist: Callable
    prior: object

    # Algorithm parameters
    n_particles: int = 1000
    v: float = 1.0
    delta: float = 0.1
    algorithm: str = "single_eps"
    resample: int | None = None

    # Proposal
    proposal: Proposal | None = None

    # RNG
    rng: np.random.Generator | None = None
    seed: int | None = None

    # Display / checkpointing
    checkpoint_history: int = 1
    show_progressbar: bool | None = None
    show_checkpoint: float | int | None = None

    def __post_init__(self):
        """Validate configuration."""
        if self.v <= 0:
            raise ValueError("Annealing speed v must be positive.")
        if self.delta <= 0:
            raise ValueError("Resampling parameter delta must be positive.")
        if self.algorithm not in ("single_eps", "multi_eps"):
            raise ValueError("algorithm must be 'single_eps' or 'multi_eps'.")
        if self.rng is not None and self.seed is not None:
            raise ValueError("Provide either rng or seed, not both.")

        _check_prior(self.prior)

        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)


# -------------------------------------------
# Result containers
# -------------------------------------------
@dataclass
class SABCState:
    """Internal algorithm state, stored alongside results."""

    epsilon: np.ndarray  # epsilon = temperature
    algorithm: str  # the algorithm used

    # Containers to store trajectories
    epsilon_history: list
    rho_history: list
    u_history: list

    # Fixed "prior-distance" reference used to build the CDF mapping
    rho_prior: np.ndarray  # prior distances computed at initialization from prior samples
    cdfs_dist_prior: CDF1D | CDFMulti  # picklable CDF mapping
    # function G in Albert et al., Statistics and Computing 25, 2015

    n_simulation: int  # number of simulations
    n_accept: int  # number of accepted updates
    n_resampling: int  # number of population resamplings
    n_population_updates: int  # number of population updates


@dataclass
class SABCResult:
    """Result of an SABC run."""

    population: np.ndarray  # parameter samples
    u: np.ndarray  # transformed distances
    rho: np.ndarray  # user-defined distances
    logprior: np.ndarray  # log prior values
    state: SABCState
    config: SABCConfig


# -------------------------------------------
# Epsilon updates
# -------------------------------------------
# Update a single epsilon.
# See eq(31) in Albert et al., Statistics and Computing 25, 2015
def update_epsilon_single_eps(u_bar: float, v: float) -> np.ndarray:
    """Compute new single epsilon from mean transformed distance.

    Args:
        u_bar: Mean of transformed distances across particles.
        v: Annealing speed parameter.

    Returns:
        Array of length 1 containing the updated epsilon.
    """
    if u_bar <= 1e-12:
        return np.array([0.0], dtype=float)

    def f(eps: float) -> float:
        return eps**2 + v * eps**1.5 - u_bar**2

    sol = root_scalar(f, bracket=(0.0, u_bar))
    if not sol.converged:
        raise RuntimeError("Failed to find root for epsilon.")
    return np.array([float(sol.root)], dtype=float)


# Update multiple epsilons.
def update_epsilon_multi_eps(u: np.ndarray, v: float) -> np.ndarray:
    """Compute new epsilon vector (one per summary statistic).

    Args:
        u: Transformed distances, shape ``(n_particles, n_stats)``.
        v: Annealing speed parameter.

    Returns:
        Array of length ``n_stats`` containing the updated epsilons.
    """
    if u.ndim != 2:
        raise ValueError("u must be 2D (n_particles, n_stats).")

    n_stats = u.shape[1]
    u_bar = np.mean(u, axis=0)  # mean over particles (vector of size n_stats)
    u_bar = np.maximum(u_bar, 1e-12)

    cn = math.factorial(2 * n_stats + 2) / (
        math.factorial(n_stats + 1) * math.factorial(n_stats + 2)
    )

    epsilon_new = np.empty(n_stats, dtype=float)

    def g(beta: float, u_bar_i: float) -> float:
        # For very small beta, use a series expansion to avoid 0/0 cancellation:
        # ratio = 1/2 - beta/12 + O(beta^2)
        if beta < 1e-6:
            return (0.5 - beta / 12.0) - u_bar_i

        # Otherwise, stable computation using expm1
        em1 = math.expm1(-beta)  # e^{-beta} - 1
        exp_neg = em1 + 1.0  # e^{-beta}

        num_g = 1.0 - exp_neg * (1.0 + beta)
        den_g = beta * (1.0 - exp_neg)

        return (num_g / den_g) - u_bar_i

    for i in range(n_stats):
        u_bar_i = float(u_bar[i])

        # A positive beta root exists only for u_bar_i < 0.5
        # (as beta -> 0, ratio -> 1/2; as beta -> inf, ratio -> 0).
        if u_bar_i >= 0.5:
            u_bar_i = 0.5 - 1e-12

        q = u_bar / u_bar_i
        num = 1.0 + np.sum(q ** (n_stats / 2))
        den = cn * (n_stats + 1) * (u_bar_i ** (1 + n_stats / 2)) * np.prod(q)

        # --- robust bracketing ---
        a = 1e-6
        b = max(1.0, 10.0 / u_bar_i)

        fa = g(a, u_bar_i)
        fb = g(b, u_bar_i)

        # Expand upper bound until sign change (should happen quickly)
        k = 0
        while fa * fb > 0 and k < 60:
            b *= 2.0
            fb = g(b, u_bar_i)
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
def resample_population(
    population: np.ndarray,
    u: np.ndarray,
    rho: np.ndarray,
    logprior: np.ndarray,
    delta: float,
    rng: np.random.Generator,
):
    """Resample population based on importance weights.

    Args:
        population: Current particle positions, shape ``(n_particles, n_para)``.
        u: Transformed distances, shape ``(n_particles, n_stats)``.
        rho: Raw distances, shape ``(n_particles, n_stats)``.
        logprior: Log prior values, shape ``(n_particles,)``.
        delta: Resampling parameter.
        rng: Random number generator.

    Returns:
        Tuple of resampled (population, u, rho, logprior, ess).
    """
    n_particles = population.shape[0]
    u_bar = np.mean(u, axis=0)
    u_bar = np.maximum(u_bar, 1e-12)

    w = np.exp(-np.sum(u * (delta / u_bar), axis=1))
    w_sum = np.sum(w)
    if w_sum == 0.0:
        raise RuntimeError("All weights are zero in resample_population.")

    p = w / w_sum
    idx = rng.choice(n_particles, size=n_particles, replace=True, p=p)

    population_resampled = population[idx, :]
    u_resampled = u[idx, :]
    rho_resampled = rho[idx, :]
    logprior_resampled = logprior[idx]

    ess = w_sum**2 / np.sum(w**2)
    return population_resampled, u_resampled, rho_resampled, logprior_resampled, ess


# -------------------------------------------
# Check if prior is valid
# -------------------------------------------
def _check_prior(prior):
    """Validate that prior has .rvs(rng) and .logpdf(theta) methods."""
    for name in ("rvs", "logpdf"):
        if not hasattr(prior, name):
            raise TypeError("prior must provide methods .rvs() and .logpdf(theta). ")
    # verify rvs accepts an rng argument
    try:
        _ = prior.rvs(np.random.default_rng(0))
    except TypeError as e:
        raise TypeError(
            "prior.rvs must accept a NumPy Generator: rvs(self, rng). "
            "Example: theta = prior.rvs(rng)"
        ) from e


# -------------------------------------------
# Extracted helpers (independently callable for profiling)
# -------------------------------------------


def _draw_prior_samples(
    f_dist: Callable,
    prior,
    n_particles: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n_particles`` from prior and evaluate f_dist for each.

    Args:
        f_dist: Distance function.
        prior: Prior distribution with ``.rvs(rng)`` and ``.logpdf(theta)``.
        n_particles: Number of particles to draw.
        rng: Random number generator.

    Returns:
        Tuple of (population, rho, logprior) arrays.
    """
    # Draw one sample to determine dimensions
    theta0 = np.asarray(prior.rvs(rng), dtype=float)
    rho0 = f_dist(theta0)
    rho0 = np.asarray(rho0, dtype=np.float64).reshape(-1)
    if rho0.ndim != 1:
        raise ValueError("f_dist must return a 1D array of distances.")
    n_para = theta0.size
    n_stats = rho0.size

    # Allocate containers
    population = np.empty((n_particles, n_para), dtype=float)
    rho = np.empty((n_particles, n_stats), dtype=float)

    # Store first sample (already computed)
    population[0, :] = theta0
    rho[0, :] = rho0

    # Fill the rest ----> CAN BE PARALLELIZED <----
    for i in range(1, n_particles):
        theta = np.asarray(prior.rvs(rng), dtype=float)
        rho_i = f_dist(theta)
        population[i, :] = theta
        rho[i, :] = np.asarray(rho_i, dtype=np.float64).reshape(-1)

    if np.any(rho < 0):
        raise ValueError("Negative distances are not allowed!")

    # Precompute log prior for current population
    logprior = np.array(
        [float(prior.logpdf(population[i, :])) for i in range(n_particles)], dtype=float
    )

    return population, rho, logprior


def _propose_and_accept(
    i: int,
    population: np.ndarray,
    u: np.ndarray,
    rho: np.ndarray,
    logprior: np.ndarray,
    pop_inactive: np.ndarray,
    proposal: Proposal,
    prior,
    f_dist: Callable,
    cdfs_dist_prior: Callable,
    inv_epsilon: np.ndarray,
    rho_buf: np.ndarray,
    u_buf: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    """Propose a new particle and accept/reject (Metropolis-Hastings step).

    Modifies ``population``, ``u``, ``rho``, ``logprior`` in-place if accepted.

    Args:
        i: Index of the particle to update.
        population: Current particle positions, shape ``(n_particles, n_para)``.
        u: Transformed distances, shape ``(n_particles, n_stats)``.
        rho: Raw distances, shape ``(n_particles, n_stats)``.
        logprior: Log prior values, shape ``(n_particles,)``.
        pop_inactive: Inactive half of the population (used by the proposal).
        proposal: Proposal mechanism.
        prior: Prior distribution.
        f_dist: Distance function.
        cdfs_dist_prior: CDF mapping function.
        inv_epsilon: Inverse of current epsilon, shape ``(n_stats,)`` or ``(1,)``.
        rho_buf: Pre-allocated buffer for proposed distances.
        u_buf: Pre-allocated buffer for proposed transformed distances.
        rng: Random number generator.

    Returns:
        True if the proposal was accepted.
    """
    theta_cur = population[i, :]
    theta_prop, log_factor = proposal(theta_cur, pop_inactive)

    lprior_prop = float(prior.logpdf(theta_prop))

    if not np.isfinite(lprior_prop):
        log_accept_prob = -np.inf
    else:
        lprior_cur = logprior[i]

        f_dist(theta_prop, out=rho_buf)
        cdfs_dist_prior(rho_buf, out=u_buf)
        log_accept_prob = (
            lprior_prop - lprior_cur + np.sum((u[i, :] - u_buf) * inv_epsilon) + log_factor
        )

    u01 = rng.random()
    if (log_accept_prob >= 0.0) or (math.log(u01) < log_accept_prob):
        population[i, :] = theta_prop
        u[i, :] = u_buf
        rho[i, :] = rho_buf
        logprior[i] = lprior_prop
        return True
    return False


def _update_single_batch(
    active: slice,
    population: np.ndarray,
    u: np.ndarray,
    rho: np.ndarray,
    logprior: np.ndarray,
    pop_inactive: np.ndarray,
    proposal: Proposal,
    prior,
    f_dist: Callable,
    cdfs_dist_prior: Callable,
    inv_epsilon: np.ndarray,
    rho_buf: np.ndarray,
    u_buf: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Update all particles in one active half-batch.

    This is the natural parallelization boundary.

    Args:
        active: Slice of particle indices to update.
        population: Current particle positions (mutated in-place).
        u: Transformed distances (mutated in-place).
        rho: Raw distances (mutated in-place).
        logprior: Log prior values (mutated in-place).
        pop_inactive: Inactive half of the population.
        proposal: Proposal mechanism.
        prior: Prior distribution.
        f_dist: Distance function.
        cdfs_dist_prior: CDF mapping function.
        inv_epsilon: Inverse of current epsilon.
        rho_buf: Pre-allocated buffer for proposed distances.
        u_buf: Pre-allocated buffer for proposed transformed distances.
        rng: Random number generator.

    Returns:
        Number of accepted proposals.
    """
    n_accept = 0
    for i in range(active.start, active.stop):
        if _propose_and_accept(
            i,
            population,
            u,
            rho,
            logprior,
            pop_inactive,
            proposal,
            prior,
            f_dist,
            cdfs_dist_prior,
            inv_epsilon,
            rho_buf,
            u_buf,
            rng,
        ):
            n_accept += 1
    return n_accept


def _record_checkpoint(
    state: SABCState,
    u: np.ndarray,
    rho: np.ndarray,
    ix: int,
    n_population_updates: int,
    t_start: int,
    checkpoint_history: int,
    show_checkpoint: float | int | None,
) -> None:
    """Record histories and log progress at checkpoint intervals.

    Args:
        state: Algorithm state to update.
        u: Current transformed distances.
        rho: Current raw distances.
        ix: Current iteration index (1-based).
        n_population_updates: Total iterations.
        t_start: Start time from ``time.perf_counter_ns()``.
        checkpoint_history: Record history every N iterations.
        show_checkpoint: Log progress every N iterations (or None to skip).
    """
    if (show_checkpoint is not None) and (ix % show_checkpoint == 0 or ix == n_population_updates):
        elapsed = (time.perf_counter_ns() - t_start) / 1e9
        eta = elapsed / ix * (n_population_updates - ix)
        eta_str = f"{eta:.2f} seconds" if eta > 1 else "< 1 second"
        LOG.debug(
            f"Update {ix}/{n_population_updates}  "
            f"avg_u={np.mean(u):.4g}  eps={np.round(state.epsilon, 4)}  ETA={eta_str}",
        )

    if ix % checkpoint_history == 0:
        state.epsilon_history.append(state.epsilon.copy())
        state.u_history.append(np.mean(u, axis=0))
        state.rho_history.append(np.mean(rho, axis=0))


# -------------------------------------------
# Initialization
# -------------------------------------------


def initialization(config: SABCConfig, n_simulation: int) -> SABCResult:
    """Initialize population from the prior.

    Draws ``n_particles`` samples from the prior, evaluates distances,
    builds CDF mapping, performs initial resampling, and computes initial epsilon.

    Args:
        config: Full SABC configuration.
        n_simulation: Total simulation budget.

    Returns:
        Initial ``SABCResult`` ready for ``update_population()``.
    """
    f_dist = config.f_dist
    prior = config.prior
    n_particles = config.n_particles
    v = config.v
    delta = config.delta
    algorithm = config.algorithm
    rng = config.rng

    if n_simulation < n_particles:
        raise ValueError(f"`n_simulation={n_simulation}` too small for {n_particles} particles.")

    LOG.info(
        f"Initialization for '{algorithm}' "
        f"with {n_particles} particles and {n_simulation} simulations."
    )

    # ---------------------
    # Draw prior samples and evaluate distances
    population, rho, logprior = _draw_prior_samples(f_dist, prior, n_particles, rng)

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
    population, u, rho_prior, logprior, ess = resample_population(
        population, u, rho_prior, logprior, delta, rng
    )

    rho_history = [np.mean(rho_prior, axis=0)]
    u_history = [np.mean(u, axis=0)]

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
        u_history=u_history,
        rho_prior=rho_prior,
        cdfs_dist_prior=cdfs_dist_prior,
        n_simulation=n_particles,
        # N.B.: we consider only n_particles draws from the prior
        # and neglect the first call to f_dist ('initialization of containers')
        n_accept=0,
        n_resampling=1,
        n_population_updates=0,
    )
    return SABCResult(
        population=population, u=u, rho=rho_prior, logprior=logprior, state=state, config=config
    )


# -------------------------------------------
# Population update
# -------------------------------------------


def update_population(
    population_state: SABCResult,
    n_simulation: int,
) -> SABCResult:
    """Update population using MCMC proposals, resampling, and annealing.

    Args:
        population_state: Current SABC result (from ``initialization()`` or a previous run).
        n_simulation: Simulation budget for this update round.

    Returns:
        Updated ``SABCResult`` (same object, mutated in-place).
    """
    config = population_state.config
    f_dist = config.f_dist
    prior = config.prior
    v = config.v
    delta = config.delta
    proposal = config.proposal
    resample = config.resample
    checkpoint_history = config.checkpoint_history
    show_progressbar = config.show_progressbar
    show_checkpoint = config.show_checkpoint
    rng = config.rng

    if show_checkpoint is None:
        show_checkpoint = 100

    # ---------------------
    # Extract variables from population_state for easier access
    # Updates inside the loop will mutate the original objects in population_state
    state = population_state.state
    population = population_state.population
    u = population_state.u
    rho = population_state.rho
    logprior = population_state.logprior
    # population, u, rho and
    # population_state.population, population_state.u, population_state.rho
    # refer to the same objects in memory, respectively.
    # Any in-place modification (e.g., population[i] = ...) affects both. Similarly for u and rho.
    # BE CAREFUL: Resampling will rebind the local names 'population' and 'u' to new objects
    # We will need to reattach them (SEE BELOW)
    n_particles = population.shape[0]
    n_stats = u.shape[1]

    # ---------------------
    # Set up proposal mechanism and resampling interval, if not provided
    if proposal is None:
        n_para = population.shape[1]
        proposal = DifferentialEvolution(n_para=n_para, rng=rng)
    if resample is None:
        resample = 2 * n_particles

    # Estimate jump covariance from current population
    update_proposal(proposal, population)

    # ---------------------
    # Each population update requires n_particles simulations
    n_population_updates = n_simulation // n_particles
    if n_population_updates == 0:
        warnings.warn(
            "n_simulation too small to perform any population update.",
            RuntimeWarning,
            stacklevel=1,
        )
        return population_state
    LOG.debug(f"Running {n_population_updates} population updates.")

    # ---------------------
    t_start = time.perf_counter_ns()

    # Buffers to avoid repeated allocations
    rho_prop_buf = np.empty(n_stats, dtype=np.float64)
    u_prop_buf = np.empty(n_stats, dtype=np.float64)

    # ---------------------
    # Main loop: at each iteration all particles are updated
    for ix in track_progress(range(1, n_population_updates + 1), show_progressbar=show_progressbar):
        inv_epsilon = 1.0 / state.epsilon

        # Split population indices in two halves
        mid = n_particles // 2
        batch_1 = slice(0, mid)
        batch_2 = slice(mid, n_particles)

        n_accept_tmp = 0
        for active, inactive in ((batch_1, batch_2), (batch_2, batch_1)):
            pop_inactive = population[inactive, :]
            n_accept_tmp += _update_single_batch(
                active,
                population,
                u,
                rho,
                logprior,
                pop_inactive,
                proposal,
                prior,
                f_dist,
                state.cdfs_dist_prior,
                inv_epsilon,
                rho_prop_buf,
                u_prop_buf,
                rng,
            )

        state.n_accept += n_accept_tmp

        # ---------------------
        # Resample population if needed
        if state.n_accept >= (state.n_resampling + 1) * resample:
            population, u, rho, logprior, ess = resample_population(
                population, u, rho, logprior, delta, rng
            )
            # Reattach new objects after resampling
            population_state.population = population
            population_state.u = u
            population_state.rho = rho
            population_state.logprior = logprior
            state.n_resampling += 1

        # ---------------------
        # Update proposal distribution and epsilon
        update_proposal(proposal, population)

        if state.algorithm == "multi_eps":
            state.epsilon = update_epsilon_multi_eps(u, v)
        else:
            state.epsilon = update_epsilon_single_eps(float(np.mean(u)), v)

        state.n_population_updates += 1
        state.n_simulation += n_particles

        _record_checkpoint(
            state, u, rho, ix, n_population_updates, t_start, checkpoint_history, show_checkpoint
        )

    # END of main loop

    # Final reattach (redundant but cheap)
    population_state.population = population
    population_state.u = u
    population_state.rho = rho
    population_state.logprior = logprior

    LOG.info(
        f"All particles have been updated {n_population_updates} times "
        f"in {(time.perf_counter_ns() - t_start) / 1e9:.2f} seconds."
    )

    return population_state


# -------------------------------------------
# Main entry point
# -------------------------------------------


def sabc(config: SABCConfig, n_simulation: int = 10_000) -> SABCResult:
    """Run the Simulated Annealing ABC algorithm.

    Args:
        config: Full SABC configuration (problem definition + algorithm parameters).
        n_simulation: Total simulation budget (initialization + population updates).

    Returns:
        ``SABCResult`` containing the posterior population and algorithm state.
    """
    # ---------------------
    # Initialization
    pop_state = initialization(config, n_simulation)

    # ---------------------
    # Sampling / Population updates
    n_sim_remaining = n_simulation - pop_state.state.n_simulation
    if n_sim_remaining < config.n_particles:
        warnings.warn(
            "`n_simulation` too small to update all particles at least once.",
            RuntimeWarning,
            stacklevel=1,
        )

    return update_population(pop_state, n_sim_remaining)
