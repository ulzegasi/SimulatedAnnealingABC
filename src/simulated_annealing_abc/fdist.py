"""fdist.py — picklable batch distance functions for SABC."""

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

DistanceMode = Literal["abs", "sq", "weighted_sq"]

SimulatorFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], None]
StatsFn = Callable[[np.ndarray, np.ndarray], None]


@dataclass
class FDist:
    """Picklable distance function — pure NumPy batch mode.

    Wraps a simulator and summary-statistics function into a single callable
    ``f_dist(theta_batch, out=None) -> np.ndarray`` that returns per-statistic
    distances for a batch of parameter vectors.

    Scratch buffers and the RNG are transient: excluded from pickle and
    recreated on deserialization so each worker process gets its own state.

    Args:
        n_samples: Size of the simulated dataset per forward-model call.
        ss_obs: Observed summary statistics (1-D).
        simulator: Batch simulator function.
            Signature: ``simulator(theta, y, rng) -> None``
            where ``theta`` has shape ``(n_batch_particles, n_para)`` and ``y`` has shape
            ``(n_batch_particles, n_samples)``.  Must fill ``y`` in-place.
        stats_fn: Batch summary-statistics function.
            Signature: ``stats_fn(y, ss) -> None``
            where ``y`` has shape ``(n_batch_particles, n_samples)`` and ``ss`` has shape
            ``(n_batch_particles, n_stats)``.  Must fill ``ss`` in-place.
        seed: RNG seed for the simulator.
        distance: Distance mode (``"abs"``, ``"sq"``, or ``"weighted_sq"``).
        weights: Weights array for ``"weighted_sq"`` distance.
    """

    n_samples: int
    ss_obs: np.ndarray
    simulator: SimulatorFn
    stats_fn: StatsFn
    seed: int | None = None
    distance: DistanceMode = "abs"
    weights: np.ndarray | None = None

    # Transient state — excluded from pickle, rebuilt in _init_buffers / _bind_transform
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)
    _y: np.ndarray = field(init=False, repr=False, compare=False)
    _ss: np.ndarray = field(init=False, repr=False, compare=False)
    _transform: Callable[[np.ndarray], None] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate inputs and initialise transient state."""
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
        self._bind_transform()

    # ------------------------------------------------------------------
    # Transient state management
    # ------------------------------------------------------------------
    def _init_buffers(self):
        """Create scratch buffers and RNG (not serialised).

        Buffers start at batch size 1 and grow lazily on first real call.
        """
        self._rng = np.random.default_rng(self.seed)
        self._y = np.empty((1, self.n_samples), dtype=np.float64)
        self._ss = np.empty((1, self.ss_obs.size), dtype=np.float64)

    def _ensure_buffers(self, n_batch_particles: int) -> None:
        """Grow scratch buffers if the current batch size exceeds capacity."""
        if self._y.shape[0] < n_batch_particles:
            self._y = np.empty((n_batch_particles, self.n_samples), dtype=np.float64)
            self._ss = np.empty((n_batch_particles, self.ss_obs.size), dtype=np.float64)

    def _bind_transform(self):
        """Bind ``_transform`` to the correct method — once, not per call."""
        if self.distance == "abs":
            self._transform = self._transform_abs
        elif self.distance == "sq":
            self._transform = self._transform_sq
        else:  # weighted_sq — already validated
            self._transform = self._transform_weighted_sq

    # ------------------------------------------------------------------
    # Pickle protocol
    # ------------------------------------------------------------------
    def __getstate__(self):
        """Exclude transient buffers and RNG from pickle."""
        state = self.__dict__.copy()
        for key in ("_rng", "_y", "_ss", "_transform"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        """Restore from pickle and recreate transient state."""
        self.__dict__.update(state)
        self._init_buffers()
        self._bind_transform()

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

        self._ensure_buffers(n_batch_particles)
        y = self._y[:n_batch_particles]
        ss = self._ss[:n_batch_particles]

        self.simulator(theta, y, self._rng)
        self.stats_fn(y, ss)

        n_stats = self.ss_obs.size
        if out is None:
            out = np.empty((n_batch_particles, n_stats), dtype=np.float64)
        else:
            if out.shape != (n_batch_particles, n_stats):
                raise ValueError(
                    f"out must have shape ({n_batch_particles}, {n_stats}), got {out.shape}."
                )
            if out.dtype != np.float64:
                raise ValueError(f"out must have dtype float64, got {out.dtype}.")

        np.subtract(ss, self.ss_obs, out=out)
        self._transform(out)
        return out

    # ------------------------------------------------------------------
    # Transform helpers (bound once via _bind_transform)
    # ------------------------------------------------------------------
    @staticmethod
    def _transform_abs(out: np.ndarray) -> None:
        np.abs(out, out=out)

    @staticmethod
    def _transform_sq(out: np.ndarray) -> None:
        np.square(out, out=out)

    def _transform_weighted_sq(self, out: np.ndarray) -> None:
        np.square(out, out=out)
        out *= self.weights


# ======================================================================
# Factory function (public API)
# ======================================================================
def make_f_dist(
    *,
    n_samples: int,
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
    """Build a picklable distance function ``f_dist(theta_batch, out=None)``.

    Pure NumPy mode (default, ``fast=False``):
      - Requires ``simulator`` and ``stats_fn``.
      - ``simulator(theta, y, rng)`` fills ``y`` in-place.
        ``theta`` has shape ``(n_batch_particles, n_para)``, ``y`` has shape
        ``(n_batch_particles, n_samples)``.
      - ``stats_fn(y, ss)`` fills ``ss`` in-place.
        ``y`` has shape ``(n_batch_particles, n_samples)``, ``ss`` has shape
        ``(n_batch_particles, n_stats)``.
      - Returns elementwise distances ``|ss - ss_obs|`` (or squared / weighted squared).

    Optional Numba mode (``fast=True``):
      - Requires ``simulator_nb`` and ``stats_fn_nb`` (both njit-compiled).
      - These operate on **single particles**: ``simulator_nb(theta, y)`` with
        ``theta`` shape ``(n_para,)`` and ``y`` shape ``(n_samples,)``.
      - The library wraps them in a parallel batch kernel automatically.
      - ``simulator`` and ``stats_fn`` are not needed and can be omitted.

    Args:
        n_samples: Size of the simulated dataset per forward-model call.
        ss_obs: Observed summary statistics (1-D).
        simulator: Batch simulator function (pure NumPy mode).
        stats_fn: Batch summary-statistics function (pure NumPy mode).
        seed: RNG seed for the simulator (pure NumPy mode only).
        distance: Distance mode (``"abs"``, ``"sq"``, or ``"weighted_sq"``).
        weights: Weights array for ``"weighted_sq"`` distance.
        fast: If ``True``, use Numba-accelerated mode.
        simulator_nb: Numba-jitted single-particle simulator (Numba mode).
        stats_fn_nb: Numba-jitted single-particle stats function (Numba mode).

    Returns:
        A callable ``f_dist(theta_batch, out=None) -> np.ndarray``.
    """
    # ---- optional Numba fast path (kept separate to keep numba truly optional)
    if fast:
        if simulator_nb is None or stats_fn_nb is None:
            raise ValueError("fast=True requires simulator_nb and stats_fn_nb.")
        from .fdist_numba import FDistNumba  # local import: optional dependency

        return FDistNumba(
            n_samples=n_samples,
            ss_obs=np.asarray(ss_obs, dtype=np.float64).reshape(-1),
            simulator_nb=simulator_nb,
            stats_fn_nb=stats_fn_nb,
            distance=distance,
            weights=weights,
        )

    # ---- validate required args for pure NumPy path
    if simulator is None:
        raise ValueError("simulator is required when fast=False (pure NumPy mode).")
    if stats_fn is None:
        raise ValueError("stats_fn is required when fast=False (pure NumPy mode).")

    return FDist(
        n_samples=n_samples,
        ss_obs=ss_obs,
        simulator=simulator,
        stats_fn=stats_fn,
        seed=seed,
        distance=distance,
        weights=weights,
    )
