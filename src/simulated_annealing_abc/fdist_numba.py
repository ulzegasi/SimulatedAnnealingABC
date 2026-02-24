"""fdist_numba.py — Numba-accelerated picklable distance function.

Single-particle ``@njit`` user functions are wrapped in a ``prange`` batch
kernel automatically, giving parallel execution over particles with no change
to user code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .fdist import DistanceMode

LOG = logging.getLogger(__name__)

try:
    import numba as nb

    _has_numba = True
except ImportError:
    _has_numba = False


# ======================================================================
# Single-particle kernel
# ======================================================================
def _build_numba_kernel(
    simulator,
    stats_fn,
    distance: DistanceMode,
):
    """Construct ``@njit`` core + transform functions for one particle.

    Args:
        simulator: Numba-jitted single-particle simulator.
        stats_fn: Numba-jitted single-particle summary-statistics function.
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
        simulator(theta, y)
        stats_fn(y, ss)
        for j in range(out.size):
            out[j] = ss[j] - ss_obs[j]
        _transform(out, w_in)

    return _core


# ======================================================================
# Batch kernel — wraps single-particle _core in nb.prange
# ======================================================================
def _build_batch_kernel(core_fn):
    """Wrap a single-particle ``_core`` in a ``prange`` batch loop.

    The resulting function processes ``n_batch_particles`` particles in parallel, each using
    its own row of the pre-allocated 2-D scratch buffers.

    Args:
        core_fn: The single-particle Numba kernel returned by
            ``_build_numba_kernel``.

    Returns:
        A Numba-jitted ``_batch(theta_2d, y_2d, ss_2d, out_2d, ss_obs, w)``
        function.
    """

    @nb.njit(parallel=True, cache=True)
    def _batch(theta_2d, y_2d, ss_2d, out_2d, ss_obs, w_in):
        n = theta_2d.shape[0]
        for i in nb.prange(n):
            core_fn(theta_2d[i], y_2d[i], ss_2d[i], out_2d[i], ss_obs, w_in)

    return _batch


# ======================================================================
# FDistNumba — public dataclass
# ======================================================================
@dataclass
class FDistNumba:
    """Picklable distance function — Numba-accelerated mode.

    The ``@njit`` kernel is built lazily on the first call and excluded from
    pickle.  Numba's on-disk cache (``cache=True``) ensures that subsequent
    rebuilds after deserialization are fast.

    User-supplied ``simulator`` and ``stats_fn`` must be ``@numba.njit``-compiled
    **single-particle** functions.  The library wraps them in a ``prange`` batch
    kernel automatically.

    Args:
        n_samples: Size of the simulated dataset per forward-model call.
        ss_obs: Observed summary statistics (1-D).
        simulator: Numba-jitted simulator. Signature: ``simulator(theta, y) -> None``
            where ``theta`` is 1-D ``(n_para,)`` and ``y`` is 1-D ``(n_samples,)``.
        stats_fn: Numba-jitted stats function. Signature: ``stats_fn(y, ss) -> None``
            where ``y`` is 1-D ``(n_samples,)`` and ``ss`` is 1-D ``(n_stats,)``.
        distance: Distance mode (``"abs"``, ``"sq"``, or ``"weighted_sq"``).
        weights: Weights array for ``"weighted_sq"`` distance.
        n_workers: Number of Numba threads for ``prange`` execution.
            Controls ``numba.set_num_threads()`` on first call.
    """

    n_samples: int
    ss_obs: np.ndarray
    simulator: Callable
    stats_fn: Callable
    distance: DistanceMode = "abs"
    weights: np.ndarray | None = None
    n_workers: int = 1

    # Transient state — excluded from pickle, rebuilt lazily
    _kernel: Callable | None = field(init=False, repr=False, compare=False, default=None)
    _batch_kernel: Callable | None = field(init=False, repr=False, compare=False, default=None)
    _y: np.ndarray = field(init=False, repr=False, compare=False)
    _ss: np.ndarray = field(init=False, repr=False, compare=False)
    _w_core: np.ndarray = field(init=False, repr=False, compare=False)
    _threads_set: bool = field(init=False, repr=False, compare=False, default=False)

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

        if self.n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {self.n_workers}.")

        self._init_buffers()

    # ------------------------------------------------------------------
    # Transient state management
    # ------------------------------------------------------------------
    def _init_buffers(self):
        """Create scratch buffers (not serialised).

        Buffers start at batch size 1 and grow lazily on first real call.
        """
        self._y = np.empty((1, self.n_samples), dtype=np.float64)
        self._ss = np.empty((1, self.ss_obs.size), dtype=np.float64)
        self._w_core = (
            self.weights
            if self.distance == "weighted_sq" and self.weights is not None
            else np.empty(1, dtype=np.float64)
        )
        self._kernel = None
        self._batch_kernel = None
        self._threads_set = False

    def _ensure_buffers(self, n_batch_particles: int) -> None:
        """Grow scratch buffers if the current batch size exceeds capacity."""
        if self._y.shape[0] < n_batch_particles:
            self._y = np.empty((n_batch_particles, self.n_samples), dtype=np.float64)
            self._ss = np.empty((n_batch_particles, self.ss_obs.size), dtype=np.float64)

    def _ensure_kernel(self):
        """Build the Numba kernels on first call (lazy JIT) and set thread count."""
        if not self._threads_set:
            nb.set_num_threads(self.n_workers)
            LOG.debug(f"Numba thread count set to {self.n_workers}.")
            self._threads_set = True
        if self._kernel is None:
            self._kernel = _build_numba_kernel(
                self.simulator,
                self.stats_fn,
                self.distance,
            )
        if self._batch_kernel is None:
            self._batch_kernel = _build_batch_kernel(self._kernel)

    # ------------------------------------------------------------------
    # Pickle protocol
    # ------------------------------------------------------------------
    def __getstate__(self):
        """Exclude transient buffers and kernel from pickle."""
        state = self.__dict__.copy()
        for key in ("_kernel", "_batch_kernel", "_y", "_ss", "_w_core", "_threads_set"):
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
            theta: Parameter batch, shape ``(n_batch_particles, n_para)``.
            out: Optional pre-allocated output buffer, shape ``(n_batch_particles, n_stats)``.

        Returns:
            Per-statistic distances, shape ``(n_batch_particles, n_stats)``.
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        n_batch_particles = theta.shape[0]
        n_stats = self.ss_obs.size

        self._ensure_buffers(n_batch_particles)
        self._ensure_kernel()

        y = self._y[:n_batch_particles]
        ss = self._ss[:n_batch_particles]

        if out is None:
            out = np.empty((n_batch_particles, n_stats), dtype=np.float64)
        else:
            if out.shape != (n_batch_particles, n_stats):
                raise ValueError(
                    f"out must have shape ({n_batch_particles}, {n_stats}), got {out.shape}."
                )
            if out.dtype != np.float64:
                raise ValueError(f"out must have dtype float64, got {out.dtype}.")

        assert self._batch_kernel is not None  # guaranteed by _ensure_kernel
        self._batch_kernel(theta, y, ss, out, self.ss_obs, self._w_core)
        return out

    # ------------------------------------------------------------------
    # Clone support (for parallel_batches in sabc.py)
    # ------------------------------------------------------------------
    def clone(self, seed: int | None = None) -> FDistNumba:
        """Create an independent copy with fresh scratch buffers.

        ``FDistNumba`` has no Python-level RNG (Numba uses thread-local
        random state inside ``prange``), so ``seed`` is accepted for API
        consistency with ``FDist.clone()`` but is unused.

        Args:
            seed: Ignored.  Accepted for API consistency with ``FDist``.

        Returns:
            A new ``FDistNumba`` with the same kernels but its own buffers.
        """
        return FDistNumba(
            n_samples=self.n_samples,
            ss_obs=self.ss_obs.copy(),
            simulator=self.simulator,
            stats_fn=self.stats_fn,
            distance=self.distance,
            weights=self.weights.copy() if self.weights is not None else None,
            n_workers=self.n_workers,
        )
