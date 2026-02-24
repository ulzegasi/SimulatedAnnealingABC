# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: sabc_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simulated Annealing ABC (SABC)
#
# ## Basic tests
# ---
#
# This notebook contains basic sanity checks for the SABC implementation,
# using a simple Gaussian toy model. The goal is to verify convergence,
# compare single- and multi-epsilon schemes, and validate posterior recovery.
#
# The observed data are generated from a normal distribution with unknown mean and standard deviation.
# Inference is performed on the parameters `(mu, sigma)` using **3 summary stats**: empirical **mean**, **standard deviation** and a **randomly generated** number. The latter is just uninformative noise.

# %%
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

from simulated_annealing_abc import (
    DifferentialEvolution,
    SABCConfig,
    make_f_dist,
    sabc,
    save_sabc_result,
    update_population,
)

# %%
# Path of *this* script
HERE = Path.cwd()


# %% [markdown]
# ### Generate data from N(mu,sigma)
# ---

# %%
# -------------------------
# Reproducibility
# -------------------------
np.random.seed(1822)

# %%
# -------------------------
# True data-generating process
# -------------------------
true_mu = 10.0
true_sigma = 15.0
n_samples = 1000

y_obs = np.random.normal(true_mu, true_sigma, size=n_samples)

# %%
# Plot the observed data and true distribution

x = np.linspace(y_obs.min(), y_obs.max(), 300)

plt.figure(figsize=(6, 4))
plt.hist(y_obs, bins=20, density=True, alpha=0.7, edgecolor="black", label="Data")
plt.plot(x, norm.pdf(x, true_mu, true_sigma), "r-", lw=2, label="True distribution")
plt.xlabel("Observed values")
plt.ylabel("Density")
plt.title("Observed data and true generating distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### True posterior
# ---

# %%
# -------------------------
# Prior definition
# -------------------------
mu_min, mu_max = -10.0, 20.0
sigma_min, sigma_max = 0.0, 25.0


# %%
# -------------------------
# Generate true posterior samples using MCMC
# -------------------------
def log_prior(theta):
    mu, sigma = theta
    if (mu_min <= mu <= mu_max) and (sigma_min <= sigma <= sigma_max):
        # uniform prior -> constant inside bounds
        return -np.log(mu_max - mu_min) - np.log(sigma_max - sigma_min)
    return -np.inf


def log_likelihood(theta, y):
    mu, sigma = theta
    if sigma <= 0:
        return -np.inf
    n = y.size
    # Normal log-likelihood
    return -n * np.log(sigma) - 0.5 * np.sum((y - mu) ** 2) / (sigma**2)


def log_posterior(theta, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y)


# --- MCMC settings ---
ndim = 2
nwalkers = 20  # number of walkers
burnin = 10000
nsteps = 5000  # production steps per walker
thin = 100

# --- initialize walkers uniformly from the prior ---
p0 = np.empty((nwalkers, ndim))
p0[:, 0] = np.random.uniform(mu_min, mu_max, size=nwalkers)  # mu
p0[:, 1] = np.random.uniform(max(sigma_min, 1e-6), sigma_max, size=nwalkers)  # sigma

# --- run MCMC ---
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(y_obs,))
sampler.run_mcmc(p0, burnin + nsteps, progress=True)

# --- discard burn-in, thin, flatten ---
flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)
mu_truepost = flat[:, 0]
sigma_truepost = flat[:, 1]

# %%
mu_truepost.shape, sigma_truepost.shape

# %%
# -------------------------
# Plot true posterior
# -------------------------

# KDE from true posterior samples
values = np.vstack([mu_truepost, sigma_truepost])
kde = gaussian_kde(values)

# Grid (manual zoom region)
mu_lims = (7, 11)
sigma_lims = (13, 17)
mu_grid = np.linspace(mu_lims[0], mu_lims[1], 250)
sig_grid = np.linspace(sigma_lims[0], sigma_lims[1], 250)
MU, SIG = np.meshgrid(mu_grid, sig_grid)

# Evaluate density
positions = np.vstack([MU.ravel(), SIG.ravel()])
Z = kde(positions).reshape(MU.shape)

plt.figure(figsize=(6, 5))
plt.grid(True, alpha=0.3, zorder=0)

# Shaded contours (grayscale)
n_levels = 10
cf = plt.contourf(MU, SIG, Z, levels=n_levels, cmap="Greys", zorder=1)

# Contour lines
plt.contour(MU, SIG, Z, levels=n_levels, colors="black", linewidths=0.6, alpha=0.6, zorder=2)

# True parameters
plt.scatter(
    true_mu, true_sigma, c="red", s=90, linewidths=3.0, marker="x", zorder=3, label="True value"
)

plt.xlim(mu_lims)
plt.ylim(sigma_lims)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\sigma$")
plt.title("True posterior")
plt.colorbar(cf, label="Posterior density")
plt.legend()
plt.show()


# %% [markdown]
# ### Setup SABC inference
# ---

# %%
# IMPORTANT -> Prior must be defined so that
# it can generate samples (with prior.rvs)
# and compute logpdf (with prior.logpdf)


class Prior:
    """Independent Uniform prior for (mu, sigma).

    Batch API:
      - ``rvs(rng, size=n_particles)`` -> ``(n_particles, 2)``
      - ``logpdf(theta_batch)`` -> ``(n_particles,)``  where ``theta_batch`` is ``(n_particles, 2)``
    """

    def rvs(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw ``size`` samples from the prior, returning shape ``(size, 2)``."""
        mu = rng.uniform(mu_min, mu_max, size=size)
        sigma = rng.uniform(sigma_min, sigma_max, size=size)
        return np.column_stack([mu, sigma])

    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate log-prior for a batch of parameters.

        Args:
            theta: Parameter array of shape ``(n_particles, 2)``.

        Returns:
            Log-prior values of shape ``(n_particles,)``.
        """
        theta = np.atleast_2d(theta)
        mu = theta[:, 0]
        sigma = theta[:, 1]
        in_bounds = (mu_min <= mu) & (mu <= mu_max) & (sigma_min <= sigma) & (sigma <= sigma_max)
        lp = np.full(theta.shape[0], -np.inf)
        lp[in_bounds] = -np.log(mu_max - mu_min) - np.log(sigma_max - sigma_min)
        return lp


# %%
prior = Prior()


# %%
# -------------------------
# Batch simulator + summary statistics (empirical mean, std, and random noise)
# -------------------------
# NOTE: Simulator and summary statistics functions fill in-place arrays
# y: observed/simulated data — shape (n_batch_particles, n_samples)
# theta: model parameters — shape (n_batch_particles, 2)
# ss_out: summary statistics output array — shape (n_batch_particles, 3)

# Module-level RNG for the noise statistic
_noise_rng = np.random.default_rng(42)


def simulator(theta: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
    """Batch simulator.

    theta (n_batch_particles, 2), y (n_batch_particles, n_samples).
    Fills y in-place.
    """
    mu = theta[:, 0:1]  # (n_batch_particles, 1)
    sigma = theta[:, 1:2]  # (n_batch_particles, 1)
    y[:] = rng.normal(loc=mu, scale=sigma, size=y.shape)


def stats_fn(y: np.ndarray, ss_out: np.ndarray) -> None:
    """Batch stats.

    y (n_batch_particles, n_samples), ss_out (n_batch_particles, 3).
    Fills ss_out in-place.
    """
    ss_out[:, 0] = np.mean(y, axis=1)
    ss_out[:, 1] = np.std(y, axis=1, ddof=0)
    ss_out[:, 2] = _noise_rng.standard_normal(y.shape[0])


# %%
# n_stats
n_stats = 3  # manually set

# compute summary statistics (ss_obs) for the observed data (y_obs)
ss_obs = np.empty((1, n_stats), dtype=np.float64)
stats_fn(y_obs.reshape(1, -1), ss_obs)
ss_obs = ss_obs.ravel()
print("Observed summary statistics:", ss_obs)
print("Number of summary statistics:", n_stats)


# %%
# -------------------------
# build allocation-free f_dist(theta, out=None)
# -------------------------
f_dist = make_f_dist(
    n_samples=n_samples,
    ss_obs=ss_obs,
    simulator=simulator,
    stats_fn=stats_fn,
    seed=123,
)


# %%
# -------------------------
# SABC parameters
# -------------------------
n_particles = 1000
n_simulation = 1_000_000
v = 1.0

# %% [markdown]
# ### Run SABC (DE, single_eps, 3 stats)
# ---

# %%
# -------------------------
# Run: Differential Evolution, Single Epsilon
# -------------------------
rng_alg = np.random.default_rng(18)
rng_prop = np.random.default_rng(22)

config_dif = SABCConfig(
    f_dist=f_dist,
    prior=prior,
    n_particles=n_particles,
    v=v,
    show_checkpoint=200,
    algorithm="single_eps",
    proposal=DifferentialEvolution(n_para=2, rng=rng_prop),
    rng=rng_alg,
)

out_dif = sabc(config_dif, n_simulation=n_simulation)

# %%
# -------------------------
# Use update_population to continue from previous result
# -------------------------
out_dif_2 = update_population(out_dif, n_simulation=n_simulation)

# %%
# Population (n_para × n_particles via transpose)
pop_dif = out_dif_2.population.T

# Epsilon history
eps_dif = np.column_stack(out_dif_2.state.epsilon_history)

# Mean rho history
rho_dif = np.column_stack(out_dif_2.state.rho_history)

# Mean u history
u_dif = np.column_stack(out_dif_2.state.u_history)

# %%
pop_dif.shape, eps_dif.shape, rho_dif.shape, u_dif.shape

# %%
# -------------------------
# Posterior sample
# -------------------------
mu = pop_dif[0, :]
sigma = pop_dif[1, :]

# %%

# -------------------------
# Compare true and sabc-inferred posteriors
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, constrained_layout=True)

# -------------------------
# LEFT: True posterior
# -------------------------
ax = axes[0]
ax.grid(True, alpha=0.3, zorder=0)

cf = ax.contourf(MU, SIG, Z, levels=n_levels, cmap="Greys", zorder=1)

ax.contour(MU, SIG, Z, levels=n_levels, colors="black", linewidths=0.6, alpha=0.6, zorder=2)

ax.scatter(
    true_mu, true_sigma, c="red", s=90, linewidths=3.0, marker="x", zorder=3, label="True value"
)

ax.set_xlim(mu_lims)
ax.set_ylim(sigma_lims)
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$\sigma$")
ax.set_title("True posterior")
ax.legend()

# -------------------------
# RIGHT: SABC posterior
# -------------------------
ax = axes[1]
ax.grid(True, alpha=0.3, zorder=0)

# KDE of SABC posterior (same style!)
values_sabc = np.vstack([mu, sigma])
kde_sabc = gaussian_kde(values_sabc)

positions = np.vstack([MU.ravel(), SIG.ravel()])
Z_sabc = kde_sabc(positions).reshape(MU.shape)

ax.contourf(MU, SIG, Z_sabc, levels=n_levels, cmap="Greys", zorder=1)

ax.contour(MU, SIG, Z_sabc, levels=n_levels, colors="black", linewidths=0.6, alpha=0.6, zorder=2)

ax.scatter(
    true_mu, true_sigma, c="red", s=90, linewidths=3.0, marker="x", zorder=3, label="True value"
)

ax.set_xlabel(r"$\mu$")
ax.set_title(r"SABC posterior")
ax.legend()

# -------------------------
# ONE shared colorbar
# -------------------------
cbar = fig.colorbar(cf, ax=axes, shrink=0.9)
cbar.set_label("Posterior density")

fig.suptitle(
    r"Proposal: Differential Evolution | Algorithm: single_eps | Summary stats: $\mu$, $\sigma$, median",
    fontsize=14,
)

plt.show()

# %%

T = eps_dif.shape[1]
it = np.arange(T)

fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)

# =====================================================
# ε
# =====================================================
# linear
ax = axes[0, 0]
ax.plot(it, eps_dif[0, :])
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"$\epsilon$")
ax.set_title(r"$\epsilon$ (linear)")

# log
ax = axes[0, 1]
ax.plot(it, eps_dif[0, :])
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"$\epsilon$")
ax.set_title(r"$\epsilon$ (log y)")

# =====================================================
# ρ
# =====================================================
# linear
ax = axes[1, 0]
ax.plot(it, rho_dif[0, :], label=r"$\mu$")
ax.plot(it, rho_dif[1, :], label=r"$\sigma$")
ax.plot(it, rho_dif[2, :], label=r"noise")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"Mean $\rho$")
ax.set_title(r"Mean $\rho$ (linear)")
ax.legend()

# log
ax = axes[1, 1]
ax.plot(it, rho_dif[0, :], label=r"$\mu$")
ax.plot(it, rho_dif[1, :], label=r"$\sigma$")
ax.plot(it, rho_dif[2, :], label=r"noise")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"Mean $\rho$")
ax.set_title(r"Mean $\rho$ (log y)")
ax.legend()

# =====================================================
# u
# =====================================================
# linear
ax = axes[2, 0]
ax.plot(it, u_dif[0, :], label=r"$\mu$")
ax.plot(it, u_dif[1, :], label=r"$\sigma$")
ax.plot(it, u_dif[2, :], label=r"noise")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"Mean $u$")
ax.set_xlabel("Iteration")
ax.set_title(r"Mean $u$ (linear)")
ax.legend()

# log
ax = axes[2, 1]
ax.plot(it, u_dif[0, :], label=r"$\mu$")
ax.plot(it, u_dif[1, :], label=r"$\sigma$")
ax.plot(it, u_dif[2, :], label=r"noise")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"Mean $u$")
ax.set_xlabel("Iteration")
ax.set_title(r"Mean $u$ (log y)")
ax.legend()

fig.suptitle(
    r"Proposal: Differential Evolution | Algorithm: single_eps | Summary stats: $\mu$, $\sigma$, noise",
    fontsize=14,
)

plt.show()

# %%
save_sabc_result(out_dif_2, HERE / "test_results" / "out_DE_sing_3statsNoise.pkl")

# -------------------------
# out_dif_2 = load_sabc_result(HERE / "test_results" / "out_DE_sing_3statsNoise.pkl")
# -------------------------

# %% [markdown]
# ### Run SABC (DE, multi_eps, 3 stats)
# ---

# %%
# -------------------------
# Run: Differential Evolution, Multi Epsilon
# -------------------------
rng_alg = np.random.default_rng(18)
rng_prop = np.random.default_rng(22)

config_dif_mult = SABCConfig(
    f_dist=f_dist,
    prior=prior,
    n_particles=n_particles,
    v=v,
    show_checkpoint=200,
    algorithm="multi_eps",
    proposal=DifferentialEvolution(n_para=2, rng=rng_prop),
    rng=rng_alg,
)

out_dif_mult = sabc(config_dif_mult, n_simulation=n_simulation)

# %%
# -------------------------
# Use update_population to continue from previous result
# -------------------------
out_dif_mult_2 = update_population(out_dif_mult, n_simulation=n_simulation)

# %%
# Population (n_para × n_particles via transpose)
pop_dif_mult = out_dif_mult_2.population.T

# Epsilon history
eps_dif_mult = np.column_stack(out_dif_mult_2.state.epsilon_history)

# Mean rho history
rho_dif_mult = np.column_stack(out_dif_mult_2.state.rho_history)

# Mean u history
u_dif_mult = np.column_stack(out_dif_mult_2.state.u_history)

# %%
pop_dif_mult.shape, eps_dif_mult.shape, rho_dif_mult.shape, u_dif_mult.shape

# %%
# -------------------------
# Posterior sample
# -------------------------
mu_mult = pop_dif_mult[0, :]
sigma_mult = pop_dif_mult[1, :]

# %%

# -------------------------
# Compare true and sabc-inferred posteriors
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, constrained_layout=True)

# -------------------------
# LEFT: True posterior
# -------------------------
ax = axes[0]
ax.grid(True, alpha=0.3, zorder=0)

cf = ax.contourf(MU, SIG, Z, levels=n_levels, cmap="Greys", zorder=1)

ax.contour(MU, SIG, Z, levels=n_levels, colors="black", linewidths=0.6, alpha=0.6, zorder=2)

ax.scatter(
    true_mu, true_sigma, c="red", s=90, linewidths=3.0, marker="x", zorder=3, label="True value"
)

ax.set_xlim(mu_lims)
ax.set_ylim(sigma_lims)
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$\sigma$")
ax.set_title("True posterior")
ax.legend()

# -------------------------
# RIGHT: SABC posterior
# -------------------------
ax = axes[1]
ax.grid(True, alpha=0.3, zorder=0)

# KDE of SABC posterior (same style!)
values_sabc_mult = np.vstack([mu_mult, sigma_mult])
kde_sabc_mult = gaussian_kde(values_sabc_mult)

positions = np.vstack([MU.ravel(), SIG.ravel()])
Z_sabc_mult = kde_sabc_mult(positions).reshape(MU.shape)

ax.contourf(MU, SIG, Z_sabc_mult, levels=n_levels, cmap="Greys", zorder=1)

ax.contour(
    MU, SIG, Z_sabc_mult, levels=n_levels, colors="black", linewidths=0.6, alpha=0.6, zorder=2
)

ax.scatter(
    true_mu, true_sigma, c="red", s=90, linewidths=3.0, marker="x", zorder=3, label="True value"
)

ax.set_xlabel(r"$\mu$")
ax.set_title(r"SABC posterior")
ax.legend()

# -------------------------
# ONE shared colorbar
# -------------------------
cbar = fig.colorbar(cf, ax=axes, shrink=0.9)
cbar.set_label("Posterior density")

fig.suptitle(
    r"Proposal: Differential Evolution | Algorithm: multi_eps | Summary stats: $\mu$, $\sigma$",
    fontsize=14,
)

plt.show()

# %%

T = eps_dif_mult.shape[1]
it = np.arange(T)

fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)

# =====================================================
# ε  (multi-eps: 3 components)
# =====================================================
# linear
ax = axes[0, 0]
ax.plot(it, eps_dif_mult[0, :], label=r"$\mu$")
ax.plot(it, eps_dif_mult[1, :], label=r"$\sigma$")
ax.plot(it, eps_dif_mult[2, :], label=r"median")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"$\epsilon$")
ax.set_title(r"$\epsilon$ (linear)")
ax.legend()

# log
ax = axes[0, 1]
ax.plot(it, eps_dif_mult[0, :], label=r"$\mu$")
ax.plot(it, eps_dif_mult[1, :], label=r"$\sigma$")
ax.plot(it, eps_dif_mult[2, :], label=r"median")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"$\epsilon$")
ax.set_title(r"$\epsilon$ (log y)")
ax.legend()

# =====================================================
# ρ
# =====================================================
# linear
ax = axes[1, 0]
ax.plot(it, rho_dif_mult[0, :], label=r"$\mu$")
ax.plot(it, rho_dif_mult[1, :], label=r"$\sigma$")
ax.plot(it, rho_dif_mult[2, :], label=r"median")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"Mean $\rho$")
ax.set_title(r"Mean $\rho$ (linear)")
ax.legend()

# log
ax = axes[1, 1]
ax.plot(it, rho_dif_mult[0, :], label=r"$\mu$")
ax.plot(it, rho_dif_mult[1, :], label=r"$\sigma$")
ax.plot(it, rho_dif_mult[2, :], label=r"median")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"Mean $\rho$")
ax.set_title(r"Mean $\rho$ (log y)")
ax.legend()

# =====================================================
# u
# =====================================================
# linear
ax = axes[2, 0]
ax.plot(it, u_dif_mult[0, :], label=r"$\mu$")
ax.plot(it, u_dif_mult[1, :], label=r"$\sigma$")
ax.plot(it, u_dif_mult[2, :], label=r"median")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"Mean $u$")
ax.set_xlabel("Iteration")
ax.set_title(r"Mean $u$ (linear)")
ax.legend()

# log
ax = axes[2, 1]
ax.plot(it, u_dif_mult[0, :], label=r"$\mu$")
ax.plot(it, u_dif_mult[1, :], label=r"$\sigma$")
ax.plot(it, u_dif_mult[2, :], label=r"median")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.set_ylabel(r"Mean $u$")
ax.set_xlabel("Iteration")
ax.set_title(r"Mean $u$ (log y)")
ax.legend()

fig.suptitle(
    r"Proposal: Differential Evolution | Algorithm: multi_eps | Summary stats: $\mu$, $\sigma$",
    fontsize=14,
)

plt.show()

# %%
save_sabc_result(out_dif_mult_2, HERE / "test_results" / "out_DE_mult_3statsNoise.pkl")

# -------------------------
# out_dif_mult_2 = load_sabc_result(HERE / "test_results" / "out_DE_mult_3statsNoise.pkl")
# -------------------------

# %%
