# cdf_estimators.py

import numpy as np
from scipy.interpolate import interp1d


def _build_cdf_1d(x):
    """
    Estimate the empirical CDF of data `x`, smoothed by interpolation.

    NOTE: This mirrors the Julia implementation and is tailored for SABC.
    Duplicates and pathological cases are not handled in a fully general way.
    """
    x = np.asarray(x, dtype=float)

    # Drop zeros (interp1d cannot handle repeated zeros)
    x = x[x > 0]

    if x.size == 0:
        # Edge case: all distances zero
        def trivial_cdf(d):
            d = np.asarray(d, dtype=float)
            return (d > 0).astype(float)

        return trivial_cdf

    # Add zero and an inflated maximum value
    a = 1.5
    values = np.concatenate(([0.0], np.sort(x), [np.max(x) * a]))

    # Corresponding probabilities
    probs = np.linspace(0.0, 1.0, num=values.size)

    # Linear interpolation with flat extrapolation
    f_interp = interp1d(
        values,
        probs,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, 1.0),
        assume_sorted=True,
    )

    def cdf(d):
        d = np.asarray(d, dtype=float)
        return f_interp(d)

    return cdf


def build_cdf(x):
    """
    Construct empirical CDF(s) for prior distances.

    Modes:
    - 1D array: returns a scalar/vector CDF
    - 2D array (n_particles, n_stats): returns a vectorized CDF
      applied component-wise
    """
    x = np.asarray(x, dtype=float)

    # 1D case
    if x.ndim == 1:
        return _build_cdf_1d(x)

    # 2D case
    if x.ndim != 2:
        raise ValueError("build_cdf expects a 1D or 2D array.")

    n_particles, n_stats = x.shape

    # Build one CDF per statistic
    cdfs = [_build_cdf_1d(x[:, j]) for j in range(n_stats)]

    def f(rho):
        rho = np.asarray(rho, dtype=float).reshape(-1)
        if rho.size != n_stats:
            raise ValueError(
                f"Expected rho of length {n_stats}, got shape {rho.shape}."
            )
        return np.array([cdfs[i](rho[i]) for i in range(n_stats)], dtype=float)

    return f