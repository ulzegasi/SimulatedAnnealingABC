"""
simulated_annealing_abc

Simulation-based inference using Simulated Annealing ABC.
"""

# Core algorithm
from .sabc import sabc, update_population, SABCResult, SABCState

# Proposal mechanisms
from .proposals import (
    Proposal,
    RandomWalk,
    DifferentialEvolution,
    StretchMove,
)

# I/O utilities
from .io import save_sabc_result, load_sabc_result

__all__ = [
    # main API
    "sabc",
    "update_population",

    # result containers
    "SABCResult",
    "SABCState",

    # proposals
    "Proposal",
    "RandomWalk",
    "DifferentialEvolution",
    "StretchMove",

    # persistence
    "save_sabc_result",
    "load_sabc_result",
]