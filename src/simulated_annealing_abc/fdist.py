"""fdist.py."""

from typing import Callable, Literal

import numpy as np

DistanceMode = Literal["abs", "sq", "weighted_sq"]

SimulatorFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], None]
StatsFn = Callable[[np.ndarray, np.ndarray], None]


def make_f_dist(
    *,
    num_samples: int,
    ss_obs: np.ndarray,
    simulator: SimulatorFn | None = None,
    stats_fn: StatsFn | None = None,
    seed: int | None = None,
    distance: DistanceMode = "abs",
    weights: np.ndarray | None = None,
    fast: bool = False,
    simulator_nb=None,
    stats_fn_nb=None,
):
    """Build an allocation-free distance function f_dist(theta, out=None).

    Pure NumPy mode (default, ``fast=False``):
      - Requires ``simulator`` and ``stats_fn``.
      - simulator(theta, y, rng) fills y in-place
      - stats_fn(y, ss) fills ss in-place
      - returns elementwise distances |ss - ss_obs| (or squared / weighted squared)

    Optional Numba mode (``fast=True``):
      - Requires ``simulator_nb`` and ``stats_fn_nb`` (both njit-compiled).
      - ``simulator`` and ``stats_fn`` are not needed and can be omitted.
      - dispatches to simulated_annealing_abc.fdist_numba.make_f_dist_numba
    """
    ss_obs = np.asarray(ss_obs, dtype=np.float64).reshape(-1)
    n_stats = ss_obs.size

    # ---- validate weights once
    w = None
    if distance == "weighted_sq":
        if weights is None:
            raise ValueError("weights must be provided when distance='weighted_sq'.")
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (n_stats,):
            raise ValueError(f"weights must have shape ({n_stats},), got {w.shape}.")

    # ---- optional Numba fast path (kept separate to keep numba truly optional)
    if fast:
        if simulator_nb is None or stats_fn_nb is None:
            raise ValueError("fast=True requires simulator_nb and stats_fn_nb.")
        from .fdist_numba import make_f_dist_numba  # local import: optional dependency

        return make_f_dist_numba(
            num_samples=num_samples,
            ss_obs=ss_obs,
            simulator_nb=simulator_nb,
            stats_fn_nb=stats_fn_nb,
            distance=distance,
            weights=w,  # pass validated/canonical weights (or None)
        )

    # ---- validate required args for pure NumPy path
    if simulator is None:
        raise ValueError("simulator is required when fast=False (pure NumPy mode).")
    if stats_fn is None:
        raise ValueError("stats_fn is required when fast=False (pure NumPy mode).")

    # ---- choose transform once (no branching per call)
    if distance == "abs":

        def transform(out: np.ndarray) -> None:
            np.abs(out, out=out)

    elif distance == "sq":

        def transform(out: np.ndarray) -> None:
            np.square(out, out=out)

    elif distance == "weighted_sq":

        def transform(out: np.ndarray) -> None:
            np.square(out, out=out)
            out *= w

    else:
        raise ValueError(f"Unknown distance='{distance}'.")

    rng = np.random.default_rng(seed)

    # ---- scratch buffers reused every call
    y = np.empty(num_samples, dtype=np.float64)
    ss = np.empty(n_stats, dtype=np.float64)

    def f_dist(theta: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D, got shape {theta.shape}")

        simulator(theta, y, rng)
        stats_fn(y, ss)

        if out is None:
            out = np.empty(n_stats, dtype=np.float64)
        else:
            if out.shape != (n_stats,):
                raise ValueError(f"out must have shape ({n_stats},), got {out.shape}.")
            if out.dtype != np.float64:
                raise ValueError(f"out must have dtype float64, got {out.dtype}.")

        np.subtract(ss, ss_obs, out=out)
        transform(out)
        return out

    return f_dist
