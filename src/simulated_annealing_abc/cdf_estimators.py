# cdf_estimators.py

import numpy as np
from scipy.interpolate import interp1d


def _build_cdf_1d(x):
    """
    Estimate the empirical CDF of data `x`, smoothed by interpolation.
    NOTE: It is not an empirical CDF in the strict statistical sense. 
    It is a monotone, interpolated approximation to the empirical CDF, 
    designed to map distances to [0,1] smoothly and robustly for use in SABC.
    This mirrors the Julia implementation and is tailored for SABC.
    
    NOTE: WIKIPEDIA def. of Empirical CDF (eCDF)
    In statistics, an empirical cumulative distribution function (eCDF) 
    is the distribution function associated with the empirical measure of a sample.
    This cumulative distribution function is a step function that jumps up by 1/n at each of the n data points. 
    Its value at any specified value of the measured variable is the fraction 
    of observations of the measured variable 
    that are less than or equal to the specified value.
    """
    # Here x is a column of the prior distance matrix
    # i.e., all distances (for all particles) for one given stat
    x = np.asarray(x, dtype=float)
    if np.any(x < 0) or not np.all(np.isfinite(x)):
        raise ValueError("build_cdf: distances must be finite and non-negative.")

    # ---------------------
    # The x-axis, including 0 and 1.5*(largest distance):
    # Drop zeros (interp1d cannot handle repeated zeros)
    x = x[x > 0]
    if x.size == 0:
        raise ValueError(
            "_build_cdf_1d: all prior distances are zero (after dropping zeros). "
            "This is almost certainly a bug or a degenerate f_dist/simulator "
            "that does not depend on theta."
        )
    # Add a single zero observation and an inflated maximum value
    a = 1.5
    x_unique = np.unique(x)  # sorted unique positive distances
    values = np.concatenate(([0.0], x_unique, [x_unique[-1] * a]))

    # ---------------------
    # The y-axis: corresponding probabilities
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
    One CDF is constructed for each statistic.
    It returns a function that applies the corresponding cdf to each statistic.
    """
    x = np.asarray(x, dtype=float)
    # build 1d CDFs
    # Here x is the 'distances_prior' matrix (n_particles, n_stats) 
    # This function selects all distances (for all particles)
    # and constructs a cdf function for each summary stat
    # 1D case
    if x.ndim == 1:
        return _build_cdf_1d(x)

    # 2D case
    if x.ndim != 2:
        raise ValueError("build_cdf expects a 1D or 2D array.")

    _, n_stats = x.shape

    # Build one CDF per statistic
    cdfs = [_build_cdf_1d(x[:, j]) for j in range(n_stats)]

    # Return a function that estimates CDF probability element-wise (for each stat, for a given particle)
    # Here rho is a vector, a row of the distance matrix, with size = number of stats
    # -> rho = distances for all stats, for ONE given particle
    def f(rho):
        rho = np.asarray(rho, dtype=float).reshape(-1)
        if rho.size != n_stats:
            raise ValueError(f"Expected rho of length {n_stats}, got {rho.size}.")
        return np.array([cdfs[i](rho[i]) for i in range(n_stats)], dtype=float)

    return f
