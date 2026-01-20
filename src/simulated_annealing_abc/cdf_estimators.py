# cdf_estimators.py

import numpy as np

def _prepare_cdf_1d(x: np.ndarray, a: float = 1.5):
    """
    Estimate the empirical CDF of data `x`, smoothed by interpolation.
    NOTE: It is not an empirical CDF in the strict statistical sense. 
    It is a monotone, interpolated approximation to the empirical CDF, 
    designed to map distances to [0,1] smoothly and robustly for use in SABC.
    
    NOTE: WIKIPEDIA def. of Empirical CDF (eCDF)
    In statistics, an empirical cumulative distribution function (eCDF) 
    is the distribution function associated with the empirical measure of a sample.
    This cumulative distribution function is a step function that jumps up by 1/n at each of the n data points. 
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


def build_cdf(x: np.ndarray, a: float = 1.5):
    """
    Construct empirical CDF(s) for prior distances.
    One CDF is constructed for each statistic.
    Returns a callable f(rho, out=None):
      - rho: shape (n_stats,)
      - out: optional preallocated array shape (n_stats,)
    """
    # Here x is the 'distances_prior' matrix (n_particles, n_stats) 
    # This function selects all distances (for all particles)
    # and constructs a cdf function for each summary stat
    x = np.asarray(x, dtype=float)
    
    # 1D case (single stat): return scalar cdf(d)
    # x is a vector of distances for one stat
    if x.ndim == 1:
        values, probs = _prepare_cdf_1d(x, a=a)

        def cdf_1d(d):
            d = np.asarray(d, dtype=float)
            return np.interp(d, values, probs, left=0.0, right=1.0)

        return cdf_1d

    # 2D case, (n_particles, n_stats) 
    if x.ndim != 2:
        raise ValueError("build_cdf expects a 1D or 2D array.")

    _, n_stats = x.shape
    
    # len(tables) = n_stats
    # Each element is a tuple (values, probs) for one stat
    tables = [ _prepare_cdf_1d(x[:, j], a=a) for j in range(n_stats) ]
    values_list = [t[0] for t in tables]
    probs_list  = [t[1] for t in tables]
    
    # rho is a row of the distance matrix, with size = number of stats
    def f(rho, out=None):
        rho = np.asarray(rho, dtype=float).reshape(-1)
        if rho.size != n_stats:
            raise ValueError(f"Expected rho of length {n_stats}, got {rho.size}.")

        if out is None:
            out = np.empty(n_stats, dtype=float)
        else:
            if out.shape != (n_stats,):
                raise ValueError(f"out must have shape ({n_stats},), got {out.shape}.")

        interp = np.interp
        vlist = values_list
        plist = probs_list

        for j in range(n_stats):
            out[j] = interp(rho[j], vlist[j], plist[j], left=0.0, right=1.0)

        return out
    
    return f
