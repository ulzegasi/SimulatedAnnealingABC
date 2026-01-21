# fdist_numba.py
from typing import Literal
import numpy as np

DistanceMode = Literal["abs", "sq", "weighted_sq"]

try:
    import numba as nb
except Exception as e:
    nb = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def make_f_dist_numba(
    *,
    num_samples: int,
    ss_obs: np.ndarray,
    simulator_nb,
    stats_fn_nb,
    distance: DistanceMode = "abs",
    weights: np.ndarray | None = None,
):
    if nb is None:
        raise ImportError("Numba is not available.") from _IMPORT_ERROR

    ss_obs = np.asarray(ss_obs, dtype=np.float64).reshape(-1)
    n_stats = ss_obs.size

    w = None
    if distance == "weighted_sq":
        if weights is None:
            raise ValueError("weights must be provided when distance='weighted_sq'.")
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (n_stats,):
            raise ValueError(f"weights must have shape ({n_stats},), got {w.shape}.")

    y = np.empty(num_samples, dtype=np.float64)
    ss = np.empty(n_stats, dtype=np.float64)

    w_core = w if distance == "weighted_sq" else np.empty(1, dtype=np.float64)

    if distance == "abs":
        @nb.njit(cache=True)
        def _transform(out, w_in):
            for j in range(out.size):
                v = out[j]
                out[j] = v if v >= 0.0 else -v

    elif distance == "sq":
        @nb.njit(cache=True)
        def _transform(out, w_in):
            for j in range(out.size):
                v = out[j]
                out[j] = v * v

    elif distance == "weighted_sq":
        @nb.njit(cache=True)
        def _transform(out, w_in):
            for j in range(out.size):
                v = out[j]
                out[j] = (v * v) * w_in[j]

    else:
        raise ValueError(f"Unknown distance='{distance}'.")

    @nb.njit(cache=True)
    def _core(theta, y, ss, out, ss_obs, w_in):
        simulator_nb(theta, y)
        stats_fn_nb(y, ss)
        for j in range(out.size):
            out[j] = ss[j] - ss_obs[j]
        _transform(out, w_in)

    def f_dist(theta: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D, got shape {theta.shape}")

        if out is None:
            out = np.empty(n_stats, dtype=np.float64)
        else:
            if out.shape != (n_stats,):
                raise ValueError(f"out must have shape ({n_stats},), got {out.shape}.")
            if out.dtype != np.float64:
                raise ValueError(f"out must have dtype float64, got {out.dtype}.")

        _core(theta, y, ss, out, ss_obs, w_core)
        return out

    return f_dist