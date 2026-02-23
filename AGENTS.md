# AGENTS.md
# Guidance for agentic coding in this repo

This repository is a Python 3.14+ codebase for Simulated Annealing ABC (SABC)
with optional Numba acceleration. There is no formal build system or test
runner; most validation happens via runnable scripts and notebooks in the
repository root.

-------------------------------------------------------------------------------
Quick commands (build / lint / test)
-------------------------------------------------------------------------------

Environment
- Create env (conda):
  - `conda env create -f environment.yml`
  - `conda activate sabc_env`
- Python target: >= 3.14 (see `environment.yml` and `README.md`).

Build
- No packaging/build command detected (no `pyproject.toml`, `setup.cfg`, etc.).
- Treat the source as importable from `src/` when running scripts.
  - Example: `python -c "import sys; sys.path.append('src')"` if needed.

Lint / format
- No linter or formatter configured in repo.
- Do not introduce new tooling unless explicitly requested.

Tests
- No pytest or unit test framework configured.
- Validation is performed via runnable scripts/notebooks in repo root.
- Example full runs:
  - `python tests_2stats.py`
  - `python tests_3stats.py`
  - `python tests_3stats_withNoise.py`
- Single test run (closest equivalent): run one script file:
  - `python tests_3stats.py`
  - `python tests_2stats.py`

Notebook parity
- The `tests_*.py` files are Jupytext paired with notebooks.
- If editing notebook logic, keep the `.py` script consistent with the
  corresponding `.ipynb` in the repository root.

-------------------------------------------------------------------------------
Code style and conventions
-------------------------------------------------------------------------------

General
- Target Python 3.14; type hints use PEP 604 (`X | Y`).
- Prefer NumPy arrays, typed with `np.ndarray`, and explicit dtypes.
- Favor allocation-free, in-place operations in performance-critical paths.

Imports
- Group imports as: standard library, third-party, local package.
- Prefer explicit imports over wildcard.
- Internal imports use absolute package paths (e.g. `simulated_annealing_abc`).
- Local/optional imports inside functions are used to avoid hard dependencies
  (e.g. numba optional path in `fdist.py`).

Formatting
- Indent with 4 spaces.
- Line length is not explicitly enforced; keep lines readable and avoid very
  long lines unless a scientific formula is clearer that way.
- Inline comments are used sparingly and only when behavior is non-obvious.

Naming
- Modules: `snake_case.py`.
- Functions and variables: `snake_case`.
- Classes and dataclasses: `CapWords`.
- Constants: `UPPER_SNAKE_CASE` (rare here; prefer local variables).
- Use descriptive names for buffers and arrays (e.g. `rho_prop_buf`).

Types and APIs
- Functions accept and return `np.ndarray` where possible.
- Accept `np.random.Generator` for RNG; if `seed` is provided, it must be used
  to construct a generator locally.
- Public API is re-exported in `src/simulated_annealing_abc/__init__.py` and
  should be kept stable.

Numerical patterns
- Use in-place NumPy operations to reduce allocations (e.g. `np.subtract` with
  `out=...`, `np.abs(out, out=out)` in `fdist.py`).
- Preallocate scratch buffers in tight loops and reuse them.
- Avoid changing array shapes inside inner loops; validate once up front.
- Use `np.isfinite` checks where log-probabilities are involved (e.g. reject
  when `logprior` is not finite).

Error handling
- Fail fast on invalid inputs with `ValueError` or `TypeError`.
- Use `RuntimeError` for algorithmic failures (e.g. root finding failures).
- Use `warnings.warn(..., RuntimeWarning)` for soft failures that still allow
  returning a partially valid result.

Logging / output
- The algorithm writes informational messages to stderr via a small helper
  (see `info(...)` in `sabc.py`).
- Prefer deterministic logs and avoid excessive prints inside inner loops.

Progress bars
- `tqdm` is used when available; otherwise fall back to plain loops.
- The decision for interactivity uses `is_interactive()` and is not
  configurable via environment variables at the moment.

Reproducibility
- There are three independent RNG streams in the system:
  1) simulator / distance
  2) SABC algorithm
  3) proposal mechanism
- Keep these independent; do not reuse a single RNG in all layers.

Algorithm-specific guidelines
- Distances (`rho`) must be non-negative; enforce and validate this invariant.
- Avoid negative or zero weights for the `weighted_sq` distance mode.
- Keep epsilon updates stable; when using series expansions, prefer numeric
  stability helpers like `math.expm1`.
- When resampling returns new arrays, reattach them to `SABCResult` to keep
  references consistent.

Optional Numba path
- Numba is optional; do not require it for baseline functionality.
- Fast path lives in `fdist_numba.py` and should accept njit-compiled functions
  that fill outputs in-place.

File locations and conventions
- Core algorithm: `src/simulated_annealing_abc/sabc.py`.
- Distance builder: `src/simulated_annealing_abc/fdist.py`.
- Proposals: `src/simulated_annealing_abc/proposals.py`.
- Persistence: `src/simulated_annealing_abc/io.py`.
- Notebook-backed tests: `tests_*.py` and `tests_*.ipynb` in repo root.

-------------------------------------------------------------------------------
Repository-specific notes for agents
-------------------------------------------------------------------------------

- No Cursor rules found (`.cursor/rules/` or `.cursorrules`).
- No GitHub Copilot instructions found (`.github/copilot-instructions.md`).
- `.env` is present but ignored by git; do not read or commit secrets.
- Use `src/` as the package root for imports when running scripts directly.

-------------------------------------------------------------------------------
When adding new code
-------------------------------------------------------------------------------

- Follow existing API patterns in `__init__.py` when exposing new public
  functions or classes.
- Maintain numpy-first semantics and avoid Python loops in hot paths unless
  necessary.
- Add type hints for public functions and data containers.
- Keep RNG handling explicit (`rng` vs `seed`); error if both are supplied.
- If introducing a new example or test, mirror the Jupytext pairing style.

-------------------------------------------------------------------------------
Suggested verification workflow for changes
-------------------------------------------------------------------------------

- Run the relevant script for your change, e.g.:
  - `python tests_2stats.py`
  - `python tests_3stats.py`
- If your change touches `make_f_dist`, run a distance-check script and verify
  the returned shape/dtype in `f_dist` for a single call.
- If your change touches proposal logic, run a short sampling run with a small
  `n_simulation` to sanity-check acceptance.
