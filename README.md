# SimulatedAnnealingABC

Approximate Bayesian Computation (ABC) is a family of simulation-based inference methods, also known as likelihood-free inference. This repository contains simulated-annealing-based ABC algorithms, collectively referred to as SABC methods.

The package includes:

- proposal mechanisms (built-in),
- the SABC algorithm itself,

and requires (user-defined):

- summary statistics
- a metric (distance),
- and a stochastic simulator.

The SABC algorithm supports both **single-ε** and **multi-ε** annealing schemes and is suitable for computationally expensive stochastic models.

---

## Features

- **Fast inner loop**
  - Allocation-free distance evaluation
  - In-place updates
  - Precomputed buffers and cached quantities
- **Reproducibility by design**
  - Independent RNG control for:
    - simulator / distance function
    - SABC algorithm (accept–reject, resampling)
    - proposal mechanisms
- **Modular architecture**
  - Plug in any simulator and summary statistics
  - Custom distance metrics (absolute, squared, weighted, …)
- **Multiple proposal mechanisms**
  - Differential Evolution
  - Random Walk
  - Stretch Move
- **Restartable runs**
  - Population updates can be continued from previous results

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ulzegasi/SimulatedAnnealingABC.git
cd SimulatedAnnealingABC
```

Create the Conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate sabc_env
```

**Note**
The code currently targets Python ≥ 3.14.

---

## Basic Usage

We consider a simple toy problem where the observed data (y_obs) are generated from a normal distribution with unknown mean and standard deviation.
Inference is performed on the parameters `(mu, sigma)` using the empirical mean and standard deviation as summary statistics.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from simulated_annealing_abc import (
    SABCConfig,
    sabc,
    make_f_dist,
    update_population,
    DifferentialEvolution,
    StretchMove,
    RandomWalk,
    save_sabc_result,
    load_sabc_result,
)
```

### 0. Generate data

```python
true_mu = 10.0
true_sigma = 15.0
np.random.seed(1822)
y_obs = np.random.normal(true_mu, true_sigma, size=1000)
```

### 1. Define a prior

The prior must provide:

- `rvs(rng)` — draw a sample using a NumPy `Generator`
- `logpdf(theta)` — compute the log-density

```python
mu_min, mu_max = -10.0, 20.0
sigma_min, sigma_max = 0.0, 25.0

class Prior:
    def rvs(self, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()
        mu = rng.uniform(mu_min, mu_max)
        sigma = rng.uniform(sigma_min, sigma_max)
        return np.array([mu, sigma])

    def logpdf(self, theta):
        mu, sigma = theta
        if mu_min <= mu <= mu_max and sigma_min <= sigma <= sigma_max:
            return -np.log(mu_max - mu_min) - np.log(sigma_max - sigma_min)
        return -np.inf

prior = Prior()
```

### 2. Define simulator and summary statistics

```python
def simulator(theta: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
    mu = float(theta[0])
    sigma = float(theta[1])
    y[:] = rng.normal(loc=mu, scale=sigma, size=y.shape[0])  # <- in-place

def stats_fn(y: np.ndarray, ss_out: np.ndarray) -> None:
    ss_out[0] = np.mean(y)
    ss_out[1] = np.std(y, ddof=0)

# function metadata:
stats_fn.n_stats = 2
```

Compute summary statistics (ss_obs) for given observed data (y_obs)

```python
n_stats = stats_fn.n_stats # infer n_stats from the function metadata
ss_obs = np.empty(n_stats, dtype=np.float64)
stats_fn(y_obs, ss_obs)
```

### 3. Build the distance function

```python
f_dist = make_f_dist(
    num_samples=1000,
    ss_obs=ss_obs,
    simulator=simulator,
    stats_fn=stats_fn,
    seed=123,          # simulator-level randomness
    distance="abs",    # distance per statistic: abs(ss_sim-ss_obs)
)
```

Available distances: "abs", "sq", "weighted_sq". Default: "abs".

### 4. Configure and run SABC

All algorithm settings are collected in a single `SABCConfig` dataclass.
The simulation budget (`n_simulation`) is the only argument passed separately,
since it typically changes between the initial run and subsequent updates.

```python
rng_alg  = np.random.default_rng(18)
rng_prop = np.random.default_rng(22)

config = SABCConfig(
    f_dist=f_dist,
    prior=prior,
    n_particles=1000,
    v=1.0,
    show_checkpoint=500,
    algorithm="single_eps",   # or "multi_eps"
    proposal=DifferentialEvolution(n_para=2, rng=rng_prop),
    rng=rng_alg,
)

result = sabc(config, n_simulation=1_000_000)
```

#### SABCConfig fields

| Field | Default | Description |
|---|---|---|
| `f_dist` | *(required)* | Distance function (from `make_f_dist` or hand-written) |
| `prior` | *(required)* | Prior with `.rvs(rng)` and `.logpdf(theta)` |
| `n_particles` | 1000 | Population size |
| `v` | 1.0 | Annealing speed |
| `delta` | 0.1 | Resampling parameter |
| `algorithm` | `"single_eps"` | `"single_eps"` or `"multi_eps"` |
| `resample` | `None` | Resampling interval (defaults to `2 * n_particles`) |
| `proposal` | `None` | Proposal mechanism (defaults to `DifferentialEvolution`) |
| `rng` | `None` | Algorithm RNG (`np.random.Generator`) |
| `seed` | `None` | Alternative to `rng` (creates one internally) |
| `checkpoint_history` | 1 | Record histories every N updates |
| `show_progressbar` | `None` | Show progress bar if available |
| `show_checkpoint` | `None` | Log progress every N updates |

### 5. Use update_population to continue from previous result

```python
result_2 = update_population(result, config, n_simulation=1_000_000)
```

### 6. Get/Plot results

Extract posterior sample (population) and trajectories for temperature (epsilon), distances (rho) and modified distances (u).

```python
# Population (n_particles × n_params)
pop = np.column_stack(result_2.population)
mu_post = pop[0, :]
sigma_post = pop[1, :]
# Epsilon trajectory
eps = np.column_stack(result_2.state.epsilon_history)
# Mean rho trajectory
rho = np.column_stack(result_2.state.rho_history)
# Mean u trajectory
u = np.column_stack(result_2.state.u_history)
```

Plot posterior sample.

```python
values = np.vstack([mu_post, sigma_post])
kde = gaussian_kde(values)
# --- Grid (manual zoom region) ---
mu_lims = (7, 11)
sigma_lims = (13, 17)
mu_grid = np.linspace(mu_lims[0], mu_lims[1], 250)
sig_grid = np.linspace(sigma_lims[0], sigma_lims[1], 250)
MU, SIG = np.meshgrid(mu_grid, sig_grid)
# --- Evaluate density ---
positions = np.vstack([MU.ravel(), SIG.ravel()])
Z = kde(positions).reshape(MU.shape)
# --- Plot ---
plt.figure(figsize=(6, 5))
plt.grid(True, alpha=0.3, zorder=0)
# --- Shaded contours (grayscale) ---
n_levels = 10
cf = plt.contourf(MU, SIG, Z, levels=n_levels, cmap="Greys", zorder=1)
# --- Contour lines ---
plt.contour(MU, SIG, Z, levels=n_levels, colors="black", linewidths=0.6, alpha=0.6, zorder=2)
# --- True parameters ---
plt.scatter(
    true_mu,
    true_sigma,
    c="red",
    s=90,
    linewidths=3.0,
    marker="x",
    zorder=3,
    label="True value"
)

plt.xlim(mu_lims)
plt.ylim(sigma_lims)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\sigma$")
plt.colorbar(cf, label="Posterior density")
plt.legend()
plt.show()
```

![SABC posterior](figures/sabc_posterior.png)

### 7. Save results

The function **save_sabc_result()** saves the full SABC result to disk using Python’s **pickle** serialization.

```python
save_sabc_result(result_2, [...some path...] / "result_2.pkl")
```

Saved results can be loaded using **load_sabc_result()**

```python
loaded_result = load_sabc_result([...some path...] / "result_2.pkl")
```

---

## Reproducibility

The algorithm has **three independent sources of randomness**:

1. **Simulator / distance function**
   - Randomness from the stochastic forward model used inside `f_dist`
   - Controlled via the `seed` argument in `make_f_dist`

2. **SABC algorithm**
   - Accept/reject decisions and population resampling
   - Controlled via `rng` or `seed` in `SABCConfig`

3. **Proposal mechanism**
   - Randomness used to generate parameter proposals
   - Each proposal object accepts its own `rng`

For **fully reproducible runs**, all three sources must be fixed explicitly.

---

## Optional Numba Acceleration

`SimulatedAnnealingABC` supports an **optional Numba-accelerated path** for the simulator and summary-statistics computation inside the distance function `f_dist`.

### Why Numba?

The dominant cost in SABC is typically the **simulator + summary statistics**.

If these components can be expressed in a Numba-compatible form (no Python objects, no dynamic allocations), they can be compiled with `numba.njit` and executed inside a tight loop.

The standard NumPy implementation remains the **default** and is often already very efficient.

Numba acceleration is therefore **optional** and intended for advanced use cases.

---

### Standard (NumPy) mode — default

```python
f_dist = make_f_dist(
    num_samples=num_samples,
    ss_obs=ss_obs,
    simulator=simulator,      # Python / NumPy
    stats_fn=stats_fn,        # Python / NumPy
)
```

This version:

- Uses NumPy throughout
- Is fully reproducible
- Requires no optional dependencies

### Numba-accelerated mode (advanced)

To enable the fast path, provide Numba-compiled versions of the simulator and summary-statistics functions and set `fast=True`. The standard `simulator` and `stats_fn` arguments can be omitted in this case:

```python
f_dist_fast = make_f_dist(
    num_samples=num_samples,
    ss_obs=ss_obs,
    fast=True,
    simulator_nb=simulator_nb,   # @numba.njit
    stats_fn_nb=stats_fn_nb,     # @numba.njit
)
```

**Requirements for Numba mode:**

- `simulator_nb(theta, y)` fills `y` in-place
- `stats_fn_nb(y, ss)` fills `ss` in-place
- Both functions must be compiled with `@numba.njit`
- Avoid per-call allocations inside the Numba functions for best performance

If Numba is not installed, attempting to use `fast=True` raises an informative error.
