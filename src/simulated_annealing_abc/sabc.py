"""sabc.py — core Simulated Annealing ABC algorithm (vectorised batch mode)."""

import logging
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import root_scalar

from .cdf_estimators import CDF1D, CDFMulti, build_cdf
from .helper import INTERACTIVE_SESSION, track_progress
from .proposals import (
    DifferentialEvolution,
    Proposal,
)

LOG = logging.getLogger(__name__)


def _format_eta_minutes(eta_seconds: float) -> str:
    """Format ETA as DD:HH:MM, rounded to the nearest minute."""
    total_minutes = max(0, int(round(eta_seconds / 60.0)))
    days, rem_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rem_minutes, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}"


# -------------------------------------------
# Configuration
# -------------------------------------------
@dataclass
class SABCConfig:
    """Full configuration for an SABC run.

    Args:
        f_dist: Batch distance function.  Either from ``make_f_dist()`` or hand-written.
            Signature: ``f_dist(theta_batch) -> np.ndarray`` with ``theta_batch`` of
            shape ``(n_batch_particles, n_para)`` and return shape ``(n_batch_particles, n_stats)``.
        prior: Prior distribution object.  Must implement:
            - ``prior.rvs(rng, size=N)`` returning ``(N, n_para)`` parameter samples.
            - ``prior.logpdf(theta_batch)`` accepting ``(N, n_para)`` and returning ``(N,)``.
        n_particles: Number of particles in the population.
        v: Annealing speed parameter (must be positive).
        delta: Resampling parameter (must be positive).
        algorithm: Epsilon update strategy, ``"single_eps"`` or ``"multi_eps"``.
        resample: Resampling interval in accepted proposals.
            Defaults to ``2 * n_particles`` if ``None``.
        proposal: Proposal mechanism (e.g. ``DifferentialEvolution``, ``RandomWalk``,
            ``StretchMove``). Defaults to ``DifferentialEvolution`` if ``None``.
        parallel_batches: If ``True``, run the two half-batch updates concurrently
            using threads (emcee-style).  Both halves see a stale snapshot of the
            other half, which changes MCMC dynamics compared to the default serial
            mode where batch 2 sees batch 1's freshly updated state.
            Requires ``proposal.clone()`` and ``f_dist.clone()`` methods.
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

    # Parallelism
    parallel_batches: bool = False

    # RNG
    rng: np.random.Generator | None = None
    seed: int | None = None

    # Display / checkpointing
    checkpoint_history: int = 1
    show_progressbar: bool | None = None
    show_checkpoint: float | int | None = 100

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

        sol = root_scalar(g, args=(u_bar_i,), bracket=(a, b), method="brentq")
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
# Check if prior is valid (batch interface)
# -------------------------------------------
def _check_prior(prior):
    """Validate that prior implements the batch interface.

    Required methods:
    - ``prior.rvs(rng, size=N)`` → ``(N, n_para)``
    - ``prior.logpdf(theta_batch)`` → ``(N,)`` where ``theta_batch`` is ``(N, n_para)``
    """
    for name in ("rvs", "logpdf"):
        if not hasattr(prior, name):
            raise TypeError(
                "prior must provide methods .rvs(rng, size=N) and .logpdf(theta_batch)."
            )

    rng = np.random.default_rng(0)

    # Check rvs(rng, size=N)
    try:
        samples = prior.rvs(rng, size=2)
    except TypeError as e:
        raise TypeError(
            "prior.rvs must accept (rng, size=N). Example: theta_batch = prior.rvs(rng, size=100)"
        ) from e
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != 2:
        raise TypeError(
            f"prior.rvs(rng, size=2) must return shape (2, n_para), got {samples.shape}"
        )

    # Check logpdf(theta_batch)
    try:
        lp = prior.logpdf(samples)
    except Exception as e:
        raise TypeError(
            "prior.logpdf must accept a 2-D array of shape (N, n_para). "
            "Example: lp = prior.logpdf(theta_batch)"
        ) from e
    lp = np.asarray(lp, dtype=float)
    if lp.shape != (2,):
        raise TypeError(
            f"prior.logpdf must return shape (N,), got {lp.shape} "
            f"for input of shape {samples.shape}"
        )


# -------------------------------------------
# Batch helpers
# -------------------------------------------


def _draw_prior_samples(
    f_dist: Callable,
    prior,
    n_particles: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n_particles`` from prior and evaluate f_dist in batch.

    Args:
        f_dist: Batch distance function.
        prior: Prior distribution with batch ``.rvs(rng, size=N)`` and ``.logpdf(theta_batch)``.
        n_particles: Number of particles to draw.
        rng: Random number generator.

    Returns:
        Tuple of (population, rho, logprior) arrays.
    """
    population = np.asarray(prior.rvs(rng, size=n_particles), dtype=float)
    if population.ndim != 2 or population.shape[0] != n_particles:
        raise ValueError(
            f"prior.rvs(rng, size={n_particles}) must return shape "
            f"({n_particles}, n_para), got {population.shape}"
        )

    rho = f_dist(population)
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim != 2 or rho.shape[0] != n_particles:
        raise ValueError(f"f_dist must return shape (n_batch_particles, n_stats), got {rho.shape}")

    if np.any(rho < 0):
        raise ValueError("Negative distances are not allowed!")

    logprior = np.asarray(prior.logpdf(population), dtype=float)
    if logprior.shape != (n_particles,):
        raise ValueError(f"prior.logpdf must return shape ({n_particles},), got {logprior.shape}")

    return population, rho, logprior


def _update_batch(
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
    """Update all particles in ``active`` slice in one vectorised pass.

    Args:
        active: Slice identifying the particles to update.
        population: Full population array, shape ``(n_particles, n_para)``.  Mutated in-place.
        u: Transformed distances, shape ``(n_particles, n_stats)``.  Mutated in-place.
        rho: Raw distances, shape ``(n_particles, n_stats)``.  Mutated in-place.
        logprior: Log prior values, shape ``(n_particles,)``.  Mutated in-place.
        pop_inactive: Frozen snapshot of the inactive half, shape ``(n_inactive, n_para)``.
        proposal: Batch proposal mechanism.
        prior: Prior distribution (batch interface).
        f_dist: Batch distance function.
        cdfs_dist_prior: Batch CDF mapping function.
        inv_epsilon: Inverse of current epsilon, shape ``(n_stats,)``.
        rho_buf: Pre-allocated scratch buffer, shape ``(>=n_batch_particles, n_stats)``.
        u_buf: Pre-allocated scratch buffer, shape ``(>=n_batch_particles, n_stats)``.
        rng: Random number generator (for accept/reject coin flips).

    Returns:
        Number of accepted proposals.
    """
    n_batch_particles = active.stop - active.start
    theta_cur = population[active]  # (n_batch_particles, n_para)

    # 1. Batch propose
    theta_prop, log_factors = proposal(
        theta_cur, pop_inactive
    )  # (n_batch_particles, n_para), (n_batch_particles,)

    # 2. Batch logprior
    lprior_prop = np.asarray(prior.logpdf(theta_prop), dtype=float)  # (n_batch_particles,)
    valid = np.isfinite(lprior_prop)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return 0

    # 3. Batch simulate + distance (only valid particles)
    rho_valid = rho_buf[:n_valid]  # (n_valid, n_stats) — scratch view
    f_dist(theta_prop[valid], out=rho_valid)

    u_valid = u_buf[:n_valid]  # (n_valid, n_stats) — scratch view
    cdfs_dist_prior(rho_valid, out=u_valid)

    # 4. Batch acceptance probability
    lprior_cur = logprior[active]  # (n_batch_particles,)
    u_cur = u[active]  # (n_batch_particles, n_stats)

    log_accept = np.full(n_batch_particles, -np.inf)

    # Map valid positions into the dense rho_valid / u_valid arrays
    u_cur_valid = u_cur[valid]  # (n_valid, n_stats)
    log_accept[valid] = (
        lprior_prop[valid]
        - lprior_cur[valid]
        + np.sum((u_cur_valid - u_valid) * inv_epsilon, axis=1)
        + log_factors[valid]
    )

    # 5. Draw uniform and compare
    u01 = rng.random(n_batch_particles)
    accepted = (log_accept >= 0.0) | (np.log(u01) < log_accept)

    # 6. Scatter accepted particles back into population arrays
    n_accepted = int(accepted.sum())
    if n_accepted == 0:
        return 0

    # Global indices of accepted particles
    active_indices = np.arange(active.start, active.stop)
    acc_idx = active_indices[accepted]

    # Map accepted → valid → dense-index in rho_valid / u_valid
    # Every accepted particle is necessarily valid (invalid ⇒ log_accept = -inf ⇒ never accepted)
    valid_cumsum = np.cumsum(valid) - 1  # (B,) — maps batch position → index in rho_valid
    acc_valid_idx = valid_cumsum[accepted]

    population[acc_idx] = theta_prop[accepted]
    logprior[acc_idx] = lprior_prop[accepted]
    rho[acc_idx] = rho_valid[acc_valid_idx]
    u[acc_idx] = u_valid[acc_valid_idx]

    return n_accepted


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
    """Record histories and optionally log progress.

    Args:
        state: Algorithm state to update.
        u: Transformed distances, shape ``(n_particles, n_stats)``.
        rho: Raw distances, shape ``(n_particles, n_stats)``.
        ix: Current iteration index.
        n_population_updates: Total number of iterations.
        t_start: Start time from ``time.perf_counter_ns()``.
        checkpoint_history: Record history every N iterations.
        show_checkpoint: Log progress every N iterations (or None to skip).
    """
    if (show_checkpoint is not None) and (ix % show_checkpoint == 0 or ix == n_population_updates):
        elapsed = (time.perf_counter_ns() - t_start) / 1e9
        eta = elapsed / ix * (n_population_updates - ix)
        eta_str = _format_eta_minutes(eta)
        message = (
            f"Update {ix}/{n_population_updates}  "
            f"avg_u={np.mean(u):.4g}  "
            f"eps={np.array2string(state.epsilon, precision=4, suppress_small=False)}  "
            f"ETA (DD:HH:MM)={eta_str}"
        )
        if INTERACTIVE_SESSION:
            LOG.debug(message)
        else:
            print(message, flush=True)

    if ix % checkpoint_history == 0:
        state.epsilon_history.append(state.epsilon.copy())
        state.u_history.append(np.mean(u, axis=0))
        state.rho_history.append(np.mean(rho, axis=0))


def _iteration_tail(
    population_state: SABCResult,
    population: np.ndarray,
    u: np.ndarray,
    rho: np.ndarray,
    logprior: np.ndarray,
    n_accept_iter: int,
    proposals: list[Proposal],
    ix: int,
    n_population_updates: int,
    t_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-iteration bookkeeping shared by serial and parallel modes.

    Handles: accept counter, conditional resampling, proposal update,
    epsilon update, state counter increments, and checkpoint recording.

    Args:
        population_state: The mutable result container.
        population: Current population array (may be rebound by resampling).
        u: Transformed distances (may be rebound by resampling).
        rho: Raw distances (may be rebound by resampling).
        logprior: Log prior values (may be rebound by resampling).
        n_accept_iter: Number of accepted proposals this iteration.
        proposals: All proposal instances to update (length 1 for serial, 3 for parallel).
        ix: Current iteration index (1-based).
        n_population_updates: Total number of iterations.
        t_start: Start time from ``time.perf_counter_ns()``.

    Returns:
        Potentially rebound (population, u, rho, logprior) arrays.
    """
    config = population_state.config
    state = population_state.state

    state.n_accept += n_accept_iter

    # Resample if needed
    resample = config.resample if config.resample is not None else 2 * config.n_particles
    if state.n_accept >= (state.n_resampling + 1) * resample:
        population, u, rho, logprior, _ess = resample_population(
            population, u, rho, logprior, config.delta, config.rng
        )
        population_state.population = population
        population_state.u = u
        population_state.rho = rho
        population_state.logprior = logprior
        state.n_resampling += 1

    # Update all proposal instances from the full population
    for prop in proposals:
        prop.update(population)

    # Update epsilon
    if state.algorithm == "multi_eps":
        state.epsilon = update_epsilon_multi_eps(u, config.v)
    else:
        state.epsilon = update_epsilon_single_eps(float(np.mean(u)), config.v)

    state.n_population_updates += 1
    state.n_simulation += config.n_particles

    _record_checkpoint(
        state,
        u,
        rho,
        ix,
        n_population_updates,
        t_start,
        config.checkpoint_history,
        config.show_checkpoint,
    )

    return population, u, rho, logprior


def _run_serial_loop(
    population_state: SABCResult,
    n_population_updates: int,
    t_start: int,
) -> None:
    """Run the main MCMC loop in serial mode.

    Batch 2 sees batch 1's freshly updated state each iteration.

    Args:
        population_state: The mutable result container.
        n_population_updates: Number of iterations to run.
        t_start: Start time from ``time.perf_counter_ns()``.
    """
    config = population_state.config
    state = population_state.state
    population = population_state.population
    u = population_state.u
    rho = population_state.rho
    logprior = population_state.logprior
    n_particles = population.shape[0]
    n_stats = u.shape[1]
    proposal = config.proposal

    mid = n_particles // 2
    batch_size = max(mid, n_particles - mid)
    rho_prop_buf = np.empty((batch_size, n_stats), dtype=np.float64)
    u_prop_buf = np.empty((batch_size, n_stats), dtype=np.float64)

    for ix in track_progress(
        range(1, n_population_updates + 1), show_progressbar=config.show_progressbar
    ):
        inv_epsilon = 1.0 / state.epsilon

        batch_1 = slice(0, mid)
        batch_2 = slice(mid, n_particles)

        n_accept_iter = 0
        for active, inactive in ((batch_1, batch_2), (batch_2, batch_1)):
            pop_inactive = population[inactive, :]
            n_accept_iter += _update_batch(
                active,
                population,
                u,
                rho,
                logprior,
                pop_inactive,
                proposal,
                config.prior,
                config.f_dist,
                state.cdfs_dist_prior,
                inv_epsilon,
                rho_prop_buf,
                u_prop_buf,
                config.rng,
            )

        population, u, rho, logprior = _iteration_tail(
            population_state,
            population,
            u,
            rho,
            logprior,
            n_accept_iter,
            [proposal],
            ix,
            n_population_updates,
            t_start,
        )


def _run_parallel_loop(
    population_state: SABCResult,
    n_population_updates: int,
    t_start: int,
) -> None:
    """Run the main MCMC loop with concurrent half-batch updates.

    Both halves see a stale snapshot of the other half (emcee-style).

    Args:
        population_state: The mutable result container.
        n_population_updates: Number of iterations to run.
        t_start: Start time from ``time.perf_counter_ns()``.
    """
    config = population_state.config
    state = population_state.state
    population = population_state.population
    u = population_state.u
    rho = population_state.rho
    logprior = population_state.logprior
    n_particles = population.shape[0]
    n_stats = u.shape[1]

    proposal = config.proposal
    f_dist = config.f_dist
    rng = config.rng

    if not hasattr(proposal, "clone"):
        raise TypeError(
            "parallel_batches=True requires proposal.clone(rng) method. "
            f"{type(proposal).__name__} does not implement clone()."
        )
    if not hasattr(f_dist, "clone"):
        raise TypeError(
            "parallel_batches=True requires f_dist.clone(seed) method. "
            f"{type(f_dist).__name__} does not implement clone()."
        )

    # Spawn independent child RNG streams for each half-batch
    parent_entropy = rng.bit_generator.seed_seq  # type: ignore[union-attr]
    child_seeds = parent_entropy.spawn(2)
    rng_1 = np.random.default_rng(child_seeds[0])
    rng_2 = np.random.default_rng(child_seeds[1])

    # Spawn independent child RNG streams for proposals
    prop_seeds = parent_entropy.spawn(2)
    proposal_1 = proposal.clone(np.random.default_rng(prop_seeds[0]))
    proposal_2 = proposal.clone(np.random.default_rng(prop_seeds[1]))

    # Spawn independent f_dist clones
    fdist_seeds = parent_entropy.spawn(2)
    f_dist_1 = f_dist.clone(seed=int(fdist_seeds[0].generate_state(1)[0]))
    f_dist_2 = f_dist.clone(seed=int(fdist_seeds[1].generate_state(1)[0]))

    mid = n_particles // 2
    batch_size = max(mid, n_particles - mid)

    # Per-half scratch buffers (no sharing between threads)
    rho_buf_1 = np.empty((batch_size, n_stats), dtype=np.float64)
    u_buf_1 = np.empty((batch_size, n_stats), dtype=np.float64)
    rho_buf_2 = np.empty((batch_size, n_stats), dtype=np.float64)
    u_buf_2 = np.empty((batch_size, n_stats), dtype=np.float64)

    try:
        pool = ThreadPoolExecutor(max_workers=2)
        for ix in track_progress(
            range(1, n_population_updates + 1), show_progressbar=config.show_progressbar
        ):
            inv_epsilon = 1.0 / state.epsilon

            batch_1 = slice(0, mid)
            batch_2 = slice(mid, n_particles)

            # Snapshot BOTH halves before either update begins
            snap_1 = population[batch_1, :].copy()
            snap_2 = population[batch_2, :].copy()

            # Submit both half-batch updates concurrently
            # batch_1 uses snap_2 as its "inactive" population (and vice versa)
            fut_1 = pool.submit(
                _update_batch,
                batch_1,
                population,
                u,
                rho,
                logprior,
                snap_2,
                proposal_1,
                config.prior,
                f_dist_1,
                state.cdfs_dist_prior,
                inv_epsilon,
                rho_buf_1,
                u_buf_1,
                rng_1,
            )
            fut_2 = pool.submit(
                _update_batch,
                batch_2,
                population,
                u,
                rho,
                logprior,
                snap_1,
                proposal_2,
                config.prior,
                f_dist_2,
                state.cdfs_dist_prior,
                inv_epsilon,
                rho_buf_2,
                u_buf_2,
                rng_2,
            )

            n_accept_iter = fut_1.result() + fut_2.result()

            population, u, rho, logprior = _iteration_tail(
                population_state,
                population,
                u,
                rho,
                logprior,
                n_accept_iter,
                [proposal_1, proposal_2, proposal],
                ix,
                n_population_updates,
                t_start,
            )
    finally:
        pool.shutdown(wait=False)


# -------------------------------------------
# Initialization
# -------------------------------------------


def initialization(config: SABCConfig, n_simulation: int) -> SABCResult:
    """Initialize population from the prior.

    Draws ``n_particles`` samples from the prior, evaluates distances in batch,
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
    # Draw prior samples and evaluate distances (batch)
    population, rho, logprior = _draw_prior_samples(f_dist, prior, n_particles, rng)

    # ------------------
    # Estimate the cdf of ρ given the prior
    rho_prior = rho.copy()
    cdfs_dist_prior = build_cdf(rho_prior)

    # Transformed distances (batch CDF call)
    u = cdfs_dist_prior(rho_prior)
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)

    # ------------------
    # Resampling before setting initial epsilon
    population, u, rho_prior, logprior, _ess = resample_population(
        population, u, rho_prior, logprior, delta, rng
    )

    rho_history = [np.mean(rho_prior, axis=0)]
    u_history = [np.mean(u, axis=0)]

    if algorithm == "multi_eps":
        epsilon = update_epsilon_multi_eps(u, v)
    elif algorithm == "single_eps":
        epsilon = update_epsilon_single_eps(float(np.mean(u)), v)
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
    """Update population using batch MCMC proposals, resampling, and annealing.

    Args:
        population_state: Current SABC result (from ``initialization()`` or a previous run).
        n_simulation: Simulation budget for this update round.

    Returns:
        Updated ``SABCResult`` (same object, mutated in-place).
    """
    config = population_state.config
    proposal = config.proposal
    rng = config.rng
    n_particles = population_state.population.shape[0]

    # ---------------------
    # Set up proposal mechanism and resampling interval, if not provided
    if proposal is None:
        n_para = population_state.population.shape[1]
        proposal = DifferentialEvolution(n_para=n_para, rng=rng)
        config.proposal = proposal
    if config.resample is None:
        config.resample = 2 * n_particles

    # Estimate jump covariance from current population
    proposal.update(population_state.population)

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
    print("Initialization done, starting population updates", flush=True)

    if config.parallel_batches:
        _run_parallel_loop(population_state, n_population_updates, t_start)
    else:
        _run_serial_loop(population_state, n_population_updates, t_start)

    # Final reattach (redundant but cheap — resampling may have rebound arrays)
    # Already handled inside _iteration_tail, but defensive.

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
