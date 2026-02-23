"""fdist_numba.py — Numba-accelerated picklable distance function."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .fdist import DistanceMode

try:
    import numba as nb

    _has_numba = True
except ImportError:
    _has_numba = False


def _build_numba_kernel(
    simulator_nb,
    stats_fn_nb,
    distance: DistanceMode,
):
    """Construct ``@njit`` core + transform functions.

    Args:
        simulator_nb: Numba-jitted simulator.
        stats_fn_nb: Numba-jitted summary-statistics function.
        distance: Distance mode.

    Returns:
        A Numba-jitted ``_core(theta, y, ss, out, ss_obs, w)`` function.
    """
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

    return _core


@dataclass
class FDistNumba:
    """Picklable distance function — Numba-accelerated mode.

    The ``@njit`` kernel is built lazily on the first call and excluded from
    pickle.  Numba's on-disk cache (``cache=True``) ensures that subsequent
    rebuilds after deserialization are fast.

    Args:
        num_samples: Size of the simulated dataset per forward-model call.
        ss_obs: Observed summary statistics (1-D).
        simulator_nb: Numba-jitted simulator. Signature: ``simulator_nb(theta, y) -> None``.
        stats_fn_nb: Numba-jitted stats function. Signature: ``stats_fn_nb(y, ss) -> None``.
        distance: Distance mode (``"abs"``, ``"sq"``, or ``"weighted_sq"``).
        weights: Weights array for ``"weighted_sq"`` distance.
    """

    num_samples: int
    ss_obs: np.ndarray
    simulator_nb: Callable
    stats_fn_nb: Callable
    distance: DistanceMode = "abs"
    weights: np.ndarray | None = None

    # Transient state — excluded from pickle, rebuilt lazily
    _kernel: Callable | None = field(init=False, repr=False, compare=False, default=None)
    _y: np.ndarray = field(init=False, repr=False, compare=False)
    _ss: np.ndarray = field(init=False, repr=False, compare=False)
    _w_core: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate inputs and initialise transient state."""
        if not _has_numba:
            raise ImportError("Numba is not available.")

        self.ss_obs = np.asarray(self.ss_obs, dtype=np.float64).reshape(-1)
        n_stats = self.ss_obs.size

        if self.distance == "weighted_sq":
            if self.weights is None:
                raise ValueError("weights must be provided when distance='weighted_sq'.")
            self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
            if self.weights.shape != (n_stats,):
                raise ValueError(f"weights must have shape ({n_stats},), got {self.weights.shape}.")
        elif self.distance not in ("abs", "sq"):
            raise ValueError(f"Unknown distance='{self.distance}'.")

        self._init_buffers()

    # ------------------------------------------------------------------
    # Transient state management
    # ------------------------------------------------------------------
    def _init_buffers(self):
        """Create scratch buffers (not serialised)."""
        self._y = np.empty(self.num_samples, dtype=np.float64)
        self._ss = np.empty(self.ss_obs.size, dtype=np.float64)
        self._w_core = (
            self.weights
            if self.distance == "weighted_sq" and self.weights is not None
            else np.empty(1, dtype=np.float64)
        )
        self._kernel = None  # rebuilt lazily on first call

    def _ensure_kernel(self):
        """Build the Numba kernel on first call (lazy JIT)."""
        if self._kernel is None:
            self._kernel = _build_numba_kernel(
                self.simulator_nb,
                self.stats_fn_nb,
                self.distance,
            )

    # ------------------------------------------------------------------
    # Pickle protocol
    # ------------------------------------------------------------------
    def __getstate__(self):
        """Exclude transient buffers and kernel from pickle."""
        state = self.__dict__.copy()
        for key in ("_kernel", "_y", "_ss", "_w_core"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        """Restore from pickle and recreate transient state."""
        self.__dict__.update(state)
        self._init_buffers()

    # ------------------------------------------------------------------
    # Call interface
    # ------------------------------------------------------------------
    def __call__(
        self,
        theta: np.ndarray,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate distance between simulated and observed summary statistics.

        Args:
            theta: Parameter vector (1-D).
            out: Optional pre-allocated output buffer.

        Returns:
            Per-statistic distances.
        """
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D, got shape {theta.shape}")

        n_stats = self.ss_obs.size
        if out is None:
            out = np.empty(n_stats, dtype=np.float64)
        else:
            if out.shape != (n_stats,):
                raise ValueError(f"out must have shape ({n_stats},), got {out.shape}.")
            if out.dtype != np.float64:
                raise ValueError(f"out must have dtype float64, got {out.dtype}.")

        self._ensure_kernel()
        assert self._kernel is not None  # guaranteed by _ensure_kernel
        self._kernel(theta, self._y, self._ss, out, self.ss_obs, self._w_core)
        return out
