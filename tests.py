"""
Simulated Annealing ABC (SABC) algorithm
Basic tests
"""

import numpy as np
from scipy.stats import norm
from pathlib import Path
from simulated_annealing_abc import (
    sabc,
    DifferentialEvolution,
    StretchMove,
    RandomWalk,
    save_sabc_result,
    load_sabc_result,
)

# Path of *this* script
HERE = Path(__file__).resolve().parent


# -------------------------
# Reproducibility
# -------------------------
np.random.seed(1111)

# -------------------------
# True data-generating process
# -------------------------
true_mu = 3.0
true_sigma = 15.0
num_samples = 100

y_obs = np.random.normal(true_mu, true_sigma, size=num_samples)

# -------------------------
# Prior definition
# -------------------------
mu_min, mu_max = -10.0, 20.0
sigma_min, sigma_max = 0.0, 25.0


# IMPORTANT -> Prior must be defined so that it can generate samples and compute logpdf
class Prior:
    """Independent Uniform prior for (mu, sigma)."""

    def rvs(self):
        mu = np.random.uniform(mu_min, mu_max)
        sigma = np.random.uniform(sigma_min, sigma_max)
        return np.array([mu, sigma], dtype=float)

    def logpdf(self, theta):
        mu, sigma = theta
        if (mu_min <= mu <= mu_max) and (sigma_min <= sigma <= sigma_max):
            return -np.log(mu_max - mu_min) - np.log(sigma_max - sigma_min)
        return -np.inf


prior = Prior()

# -------------------------
# Summary statistics
# -------------------------
def sum_stats(data):
    stat1 = np.mean(data)
    stat2 = np.std(data, ddof=0)
    return np.array([stat1, stat2], dtype=float)


ss_obs = sum_stats(y_obs)
n_stats = ss_obs.size

# -------------------------
# Model + distance
# -------------------------
def model(theta):
    mu, sigma = theta
    y = np.random.normal(mu, sigma, size=num_samples)
    return sum_stats(y)


def f_dist(theta):
    ss = model(theta)
    rho = np.abs(ss - ss_obs)   # Euclidean in 1D == abs
    return rho

# -------------------------
# SABC parameters
# -------------------------
n_particles = 1000
n_simulation = 5_000_000
v = 1.0

# -------------------------
# Run: Differential Evolution
# -------------------------
out_dif = sabc(
    f_dist,
    prior,
    n_particles=n_particles,
    n_simulation=n_simulation,
    v=v,
    algorithm="single_eps",
    proposal=DifferentialEvolution(n_para=2),
)

save_sabc_result(out_dif, HERE / "test_results" / "out_differential_evolution.pkl")

# -------------------------
# Run: Stretch Move
# -------------------------
out_str = sabc(
    f_dist,
    prior,
    n_particles=n_particles,
    n_simulation=n_simulation,
    v=v,
    algorithm="single_eps",
    proposal=StretchMove(),
)

save_sabc_result(out_str, HERE / "test_results" / "out_stretch_move.pkl")

# -------------------------
# Run: Random Walk
# -------------------------
out_rnd = sabc(
    f_dist,
    prior,
    n_particles=n_particles,
    n_simulation=n_simulation,
    v=v,
    algorithm="single_eps",
    proposal=RandomWalk(n_para=2),
)

save_sabc_result(out_rnd, HERE / "test_results" / "out_random_walk.pkl")

print("All runs completed and saved.")