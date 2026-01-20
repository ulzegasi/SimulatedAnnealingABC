import numpy as np
from typing import Callable, Literal

DistanceMode = Literal["abs", "sq", "weighted_sq"]

# Type hints:
# simulator(theta, y, rng) -> None     (fills y in-place)
# stats_fn(y, ss_out) -> None          (fills ss_out in-place)

def make_f_dist(
    *,
    num_samples: int,
    ss_obs: np.ndarray,
    simulator: Callable[[np.ndarray, np.ndarray, np.random.Generator], None],
    stats_fn: Callable[[np.ndarray, np.ndarray], None],
    seed: int | None = None,
    distance: DistanceMode = "abs",
    weights: np.ndarray | None = None,
):
    """
    Build an allocation-free f_dist(theta, out=None) for SABC.

    Parameters
    ----------
    num_samples : int
        Length of simulated data vector y.
    ss_obs : (n_stats,) array
        Observed summary statistics.
    simulator :
        Function that generates simulated data in-place:
            simulator(theta, y, rng) -> None
    stats_fn :
        Function that computes summary statistics in-place:
            stats_fn(y, ss_out) -> None
    seed : int or None
        RNG seed.
    distance : {"abs", "sq", "weighted_sq"}
        Distance metric to use.
    weights : (n_stats,) array or None
        Weights for "weighted_sq" distance. Required if distance="weighted_sq".

    Returns
    -------
    f_dist(theta, out=None) -> out, where out has shape (n_stats,)
    """
    ss_obs = np.asarray(ss_obs, dtype=np.float64).reshape(-1)
    n_stats = ss_obs.size
    
    # Validate / prepare weights (only once)
    w = None
    if distance == "weighted_sq":
        if weights is None:
            raise ValueError("weights must be provided when distance='weighted_sq'.")
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (n_stats,):
            raise ValueError(f"weights must have shape ({n_stats},), got {w.shape}.")

    # Pick the distance transform once (no branching inside inner loop)
    if distance == "abs":
        def compute(out: np.ndarray) -> None:
            np.abs(out, out=out)
    elif distance == "sq":
        def compute(out: np.ndarray) -> None:
            np.square(out, out=out)
    elif distance == "weighted_sq":
        def compute(out: np.ndarray) -> None:
            np.square(out, out=out)
            out *= w  # w is a captured float64 array
    else:
        raise ValueError(f"Unknown distance='{distance}'.")

    rng = np.random.default_rng(seed)

    # Scratch buffers reused every call
    y = np.empty(num_samples, dtype=np.float64)
    ss = np.empty(n_stats, dtype=np.float64)

    def f_dist(theta: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D, got shape {theta.shape}")

        # simulate in-place
        simulator(theta, y, rng)

        # stats in-place
        stats_fn(y, ss)

        # distances in-place
        if out is None:
            out = np.empty(n_stats, dtype=np.float64)
        else:
            if out.shape != (n_stats,):
                raise ValueError(f"out must have shape ({n_stats},), got {out.shape}.")
            if out.dtype != np.float64:
                raise ValueError(f"out must have dtype float64, got {out.dtype}.")

        np.subtract(ss, ss_obs, out=out)
        compute(out)
        return out

    return f_dist