# AGENTS.md

Guidelines for AI agents working in the SimulatedAnnealingABC repository.

## Quick Start

**Note:** Run `make env-dev` first to create the development environment.

```bash
make env-dev          # Create dev environment (recommended, includes ruff, ty, pytest, etc.)
make test             # Run fast tests
make lint             # Lint code
make format           # Format code
```

For venv/system Python, use `make install-dev` instead (fewer dev tools pre-installed).

## Build & Development Setup

### Devcontainer (Recommended)

The devcontainer automatically sets up the sabc-dev environment with all dependencies and tools.
The post-create command also installs:

-   npm packages: `opencode-ai`, `markdownlint-cli2`, formatters
-   Symlinks opencode auth.json if available

### Manual Setup

```bash
make env-dev          # Create conda dev environment (includes all dependencies)
```

## Development Commands

|  Command             |  Description                           |
| -------------------- | -------------------------------------- |
|  `make env`          |  Create runtime environment            |
|  `make env-dev`      |  Create dev environment (recommended)  |
|  `make install`      |  Install package with pip (editable)   |
|  `make install-dev`  |  Install with pip + all optional deps  |
|  `make lint`         |  Lint code with ruff                   |
|  `make lint-fix`     |  Auto-fix lint issues                  |
|  `make format`       |  Format code                           |
|  `make lint-md`      |  Lint markdown files                   |
|  `make lint-md-fix`  |  Auto-fix markdown issues              |
|  `make typecheck`    |  Type check with ty                    |
|  `make test`         |  Run fast tests                        |
|  `make test-all`     |  Run all tests                         |
|  `make test-slow`    |  Run integration tests                 |
|  `make test-cov`     |  Run tests with coverage               |
|  `make test-html`    |  Generate HTML coverage report         |
|  `make notebooks`    |  Regenerate .ipynb from .py files      |
|  `make clean`        |  Clean up cache files                  |
|  `make clean-cache`  |  Clean pip, micromamba, npm caches     |

## Dependency Management

|  File                        |  Purpose                             |
| ---------------------------- | ------------------------------------ |
|  `pyproject.toml`            |  pip dependencies (source of truth)  |
|  `environment.sabc.yml`      |  Conda runtime environment           |
|  `environment.sabc-dev.yml`  |  Conda dev environment               |

When adding a new dependency:

1.  Add to **pyproject.toml** (source of truth)
2.  Add to **environment.sabc.yml** (conda runtime)
3.  Add dev-only tools to **environment.sabc-dev.yml**

## Code Style

### Python Version & RNG

-   Python >= 3.10
-   Use `np.random.Generator` everywhere (never legacy `np.random.RandomState`)
-   Use `np.random.default_rng(seed)` when creating generators

### Formatting

-   **Line length: 100** (enforced by ruff)
-   **4-space indentation** (standard Python)
-   Minimal comments — only for non-obvious logic
-   No `print()` in library code

### Imports

Standard library, then third-party, then local. Ruff enforces isort ordering (`I` rule).
Use **relative imports** within the `src/simulated_annealing_abc/` package.

```python
import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import root_scalar

from .cdf_estimators import build_cdf
from .helper import track_progress
from .proposals import DifferentialEvolution, Proposal
```

Test files use **absolute imports**: `from simulated_annealing_abc import ...`.

### Type Hints

-   Modern union syntax: `X | None` (not `Optional[X]`)
-   Use `np.ndarray` for array types, `np.float64` for explicit dtypes
-   Type aliases for callables:
  `SimulatorFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], None]`
-   All function signatures must have type hints:

```python
def sabc(
    config: SABCConfig,
    n_simulation: int = 10_000,
) -> SABCResult:
```

### Naming

-   Classes: `PascalCase` — `SABCResult`, `SABCConfig`, `DifferentialEvolution`, `StretchMove`
-   Functions/variables: `snake_case` — `update_population`, `resample_population`
-   Constants: `UPPER_SNAKE_CASE` — `LOG`
-   Private: leading underscore — `_check_prior`, `_prepare_cdf_1d`

### Error Handling

-   `ValueError` for invalid arguments
-   `TypeError` for wrong types
-   `RuntimeError` for algorithmic failures (convergence, etc.)
-   Always include variable values in messages:

```python
raise ValueError(f"`n_simulation={n_simulation}` too small for {n_particles} particles.")
```

### Logging

Use module-level logger: `LOG = logging.getLogger(__name__)`.
Use `LOG.info()`, `LOG.debug()`, `LOG.warning()`. No `print()` in library code.

### Docstrings

Google-style (enforced by ruff `D` rule with `convention = "google"`):

```python
def resample_population(
    population: np.ndarray, rho: np.ndarray, delta: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample population based on importance weights.

    Args:
        population: Current particle positions.
        rho: User-defined distances.
        delta: Resampling parameter.
        rng: Random number generator.

    Returns:
        Tuple of resampled arrays and ESS.
    """
```

### Numerical Performance

-   In-place array operations: `y[:] = rng.normal(...)`, `np.abs(out, out=out)`
-   Preallocate buffers; avoid allocations in hot loops
-   Numba acceleration is optional (pass `use_numba=True` to `make_f_dist`)

### Data Classes

`@dataclass` for result containers and config; `@dataclass(init=False)` for proposals
with custom `__init__`.

### Jupyter Notebooks

The `.py` files in `examples/notebooks/` are paired with Jupyter notebooks via jupytext's `percent` format.
**The `.py` files are the source of truth** — never manually edit `.ipynb` files.

```bash
make notebooks   # Regenerate .ipynb files from .py sources
```

## Architecture

### Package Layout (`src/simulated_annealing_abc/`)

|  Module               |  Purpose                                                                                                                                                                      |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  `sabc.py`            |  Core algorithm: `sabc()`, `update_population()`, `SABCConfig`, `SABCResult`, `SABCState`. Supports `parallel_batches` for concurrent half-batch updates.                     |
|  `proposals.py`       |  `Proposal`, `RandomWalk`, `DifferentialEvolution`, `StretchMove`. Each has a `clone(rng)` method for creating independent copies.                                            |
|  `fdist.py`           |  `FDist`, `make_f_dist()` — builds allocation-free distance functions. Supports `n_workers` for multi-threaded simulator execution and `clone(seed)` for independent copies.  |
|  `fdist_numba.py`     |  Numba-accelerated variant (lazy-loaded when `use_numba=True`). Has `clone(seed)` for API consistency.                                                                        |
|  `cdf_estimators.py`  |  `build_cdf()` for empirical CDF construction                                                                                                                                 |
|  `io.py`              |  `save_sabc_result()`, `load_sabc_result()` (pickle)                                                                                                                          |
|  `helper.py`          |  Progress bar utilities                                                                                                                                                       |

### Key Interfaces

-   **Prior** — any object with `.rvs(rng, size=)` and `.logpdf(theta)` methods.
-   **Simulator/stats_fn** — must fill output arrays in-place:
  `simulator(theta, y, rng) -> None`, `stats_fn(y, ss) -> None`.
-   **f_dist** — distance function built by `make_f_dist()` or hand-written.
  Signature: `f_dist(theta, out=None) -> np.ndarray`.
-   **clone** — `f_dist.clone(seed)` and `proposal.clone(rng)` create independent copies
  with their own RNG streams and buffers. Required by `parallel_batches=True`.

### Parallelism

Two composable layers, both using `ThreadPoolExecutor`:

1.  **`n_workers`** (on `FDist` / `make_f_dist`) — splits each simulator batch across
   N threads. In NumPy mode, each worker gets its own RNG stream (via
   `SeedSequence.spawn`) and scratch buffers. In Numba mode (`use_numba=True`),
   controls `nb.set_num_threads()` for Numba's `prange` thread pool.
2.  **`parallel_batches`** (on `SABCConfig`) — runs the two half-population updates
   concurrently (emcee-style). Clones the proposal and f_dist into 2 independent
   instances. Changes MCMC dynamics (both halves see stale snapshots).

Both default to off (serial). Both compose: `n_workers=4` + `parallel_batches=True`
= up to 8 threads per update.

**Thread-safety:** `prior.logpdf()` must be stateless (called from worker threads
when `parallel_batches=True`). `simulator` and `stats_fn` must only write to their
provided output arrays (called from worker threads when `n_workers > 1`).

**Oversubscription:** Total threads = `2 * n_workers` when both layers are active.
Keep `n_workers` at or below physical core count; reduce when combining with
`parallel_batches`. NumPy BLAS threads (`OMP_NUM_THREADS`) compound the issue.

### Reproducibility

Three independent RNG streams: simulator, algorithm, proposal.
Each accepts its own `rng` or `seed` parameter.
When `parallel_batches=True`, child RNG streams are spawned from the parent via
`SeedSequence.spawn()`, ensuring independence.
