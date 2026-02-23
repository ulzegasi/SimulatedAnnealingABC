"""cdf_estimators.py — picklable empirical CDF mappings for SABC."""

from dataclasses import dataclass

import numpy as np


def _prepare_cdf_1d(x: np.ndarray, a: float = 1.5):
    """Estimate the empirical CDF of data `x`, smoothed by interpolation.

    NOTE: It is not an empirical CDF in the strict statistical sense.
    It is a monotone, interpolated approximation to the empirical CDF,
    designed to map distances to [0,1] smoothly and robustly for use in SABC.

    NOTE: WIKIPEDIA def. of Empirical CDF (eCDF)
    In statistics, an empirical cumulative distribution function (eCDF)
    is the distribution function associated with the empirical measure of a sample.
    This cumulative distribution function is a step function that jumps up by
    1/n at each of the n data points.
    Its value at any specified value of the measured variable is the fraction
    of observations of the measured variable that are less than or equal to the specified value.

    The function: prepare (values, probs) arrays for a monotone interpolated CDF mapping.
    - Drops zeros (to avoid repeated zeros).
    - Adds one explicit 0 and an inflated maximum (a * max).
    - Builds a uniform prob grid in [0,1].
    """
    # Here x is a column of the prior distance matrix
    # i.e., all distances (for all particles) for one given stat
    x = np.asarray(x, dtype=float)
    if np.any(x < 0) or not np.all(np.isfinite(x)):
        raise ValueError("build_cdf: distances must be finite and non-negative.")

    # ---------------------
    # The x-axis, including 0 and 1.5*(largest distance):
    # Drop zeros (interp1d cannot handle repeated zeros)
    x = x[x > 0.0]
    if x.size == 0:
        raise ValueError(
            "_prepare_cdf_1d: all prior distances are zero (after dropping zeros). "
            "This likely indicates a degenerate f_dist/simulator."
        )
    # Add a single zero observation and an inflated maximum value
    x_unique = np.unique(x)  # sorted unique positive distances
    values = np.concatenate(([0.0], x_unique, [x_unique[-1] * a]))

    # ---------------------
    # The y-axis: corresponding probabilities
    probs = np.linspace(0.0, 1.0, num=values.size)

    return values, probs


# ======================================================================
# Picklable callable classes
# ======================================================================


@dataclass
class CDF1D:
    """Picklable CDF mapping for a single summary statistic.

    All fields are plain NumPy arrays — natively picklable.

    Args:
        values: Sorted distance knots for interpolation.
        probs: Corresponding probability knots in [0, 1].
    """

    values: np.ndarray
    probs: np.ndarray

    def __call__(self, d) -> np.ndarray:
        """Evaluate the CDF at distance(s) ``d``.

        Args:
            d: Scalar or array of distances.

        Returns:
            CDF values in [0, 1].
        """
        d = np.asarray(d, dtype=float)
        return np.interp(d, self.values, self.probs, left=0.0, right=1.0)


@dataclass
class CDFMulti:
    """Picklable CDF mapping for multiple summary statistics.

    Stores one interpolation table per statistic. All fields are plain
    lists of NumPy arrays — natively picklable.

    Args:
        values_list: Per-statistic sorted distance knots.
        probs_list: Per-statistic probability knots in [0, 1].
        n_stats: Number of summary statistics.
    """

    values_list: list[np.ndarray]
    probs_list: list[np.ndarray]
    n_stats: int

    def __call__(self, rho: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Evaluate CDF for each statistic.

        Args:
            rho: Distance vector of shape ``(n_stats,)``.
            out: Optional pre-allocated output buffer of shape ``(n_stats,)``.

        Returns:
            CDF values in [0, 1] for each statistic.
        """
        rho = np.asarray(rho, dtype=float).reshape(-1)
        if rho.size != self.n_stats:
            raise ValueError(f"Expected rho of length {self.n_stats}, got {rho.size}.")

        if out is None:
            out = np.empty(self.n_stats, dtype=float)
        elif out.shape != (self.n_stats,):
            raise ValueError(f"out must have shape ({self.n_stats},), got {out.shape}.")

        interp = np.interp
        vlist = self.values_list
        plist = self.probs_list

        for j in range(self.n_stats):
            out[j] = interp(rho[j], vlist[j], plist[j], left=0.0, right=1.0)

        return out


# ======================================================================
# Factory function (public API, backward-compatible)
# ======================================================================


def build_cdf(x: np.ndarray, a: float = 1.5) -> CDF1D | CDFMulti:
    """Construct picklable empirical CDF(s) for prior distances.

    One CDF is constructed for each statistic.
    Returns a callable ``f(rho, out=None)``:
      - rho: shape ``(n_stats,)``
      - out: optional preallocated array shape ``(n_stats,)``

    Args:
        x: Distance array — 1-D for a single statistic, or 2-D ``(n_particles, n_stats)``.
        a: Inflation factor for the maximum distance knot (default 1.5).

    Returns:
        ``CDF1D`` for a single statistic, ``CDFMulti`` for multiple statistics.
    """
    x = np.asarray(x, dtype=float)

    # 1D case (single stat): return scalar cdf(d)
    if x.ndim == 1:
        values, probs = _prepare_cdf_1d(x, a=a)
        return CDF1D(values=values, probs=probs)

    # 2D case, (n_particles, n_stats)
    if x.ndim != 2:
        raise ValueError("build_cdf expects a 1D or 2D array.")

    _, n_stats = x.shape

    tables = [_prepare_cdf_1d(x[:, j], a=a) for j in range(n_stats)]
    values_list = [t[0] for t in tables]
    probs_list = [t[1] for t in tables]

    return CDFMulti(values_list=values_list, probs_list=probs_list, n_stats=n_stats)
