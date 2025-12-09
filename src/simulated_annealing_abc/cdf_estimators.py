# cdf_estimators.py

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from scipy.interpolate import interp1d


def _build_cdf_1d(x: ArrayLike) -> Callable[[float | ArrayLike], np.ndarray]:
    """
    Estimate the empirical CDF of data `x`, smoothed by interpolation.

    NOTE: This is a direct analogue of the Julia function in cdf_estimators.jl
    and inherits the same caveat: it is not a general-purpose ECDF implementation,
    and duplicates in x are not treated carefully. For SABC use, this is fine.

    Returns
    -------
    f : callable
        f(d) returns values in [0, 1] for scalar or array-like d.
    """
    x = np.asarray(x, dtype=float)

    # Drop zeros (as in the Julia version: interpolate cannot handle multiple 0s)
    x = x[x > 0]

    if x.size == 0:
        # Edge case: if all distances were zero, we fall back to a trivial CDF
        # that is zero for d <= 0 and 1 for d > 0.
        def trivial_cdf(d: ArrayLike) -> np.ndarray:
            d = np.asarray(d, dtype=float)
            return (d > 0).astype(float)
        return trivial_cdf

    # Add single zero observation and a larger max observation
    a = 1.5
    values = np.concatenate(([0.0], np.sort(x), [np.max(x) * a]))

    # y-axis: probabilities 0..1
    probs = np.linspace(0.0, 1.0, num=values.size)

    # Linear interpolation, flat extrapolation outside the observed range
    f_interp = interp1d(
        values,
        probs,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, 1.0),  # below min -> 0, above max -> 1
        assume_sorted=True,
    )

    def cdf(d: ArrayLike) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        return f_interp(d)

    return cdf


def build_cdf(x: ArrayLike) -> Callable[[ArrayLike], np.ndarray]:
    """
    Construct empirical CDF(s) for prior distances.

    Two modes, mirroring the Julia implementation:

    1) x is 1D (n_samples,):
       Returns a CDF function f(d) -> array in [0,1].

    2) x is 2D (n_particles, n_stats):
       Returns a function f(rho) that applies the corresponding
       1D CDF to each component:
          rho: 1D array of length n_stats (distances for one particle)
          -> 1D array of transformed distances in [0,1].
    """
    x = np.asarray(x, dtype=float)

    # 1D case
    if x.ndim == 1:
        return _build_cdf_1d(x)

    # 2D case: (n_particles, n_stats)
    if x.ndim != 2:
        raise ValueError("build_cdf expects a 1D or 2D array.")

    n_particles, n_stats = x.shape

    # Build 1D CDFs for each statistic (column)
    cdfs = [_build_cdf_1d(x[:, j]) for j in range(n_stats)]

    def f(rho: ArrayLike) -> np.ndarray:
        rho = np.asarray(rho, dtype=float).reshape(-1)
        if rho.size != n_stats:
            raise ValueError(
                f"Expected rho of length {n_stats}, got shape {rho.shape}."
            )
        # Apply each cdf to the corresponding component
        return np.array([cdfs[i](rho[i]) for i in range(n_stats)], dtype=float)

    return f