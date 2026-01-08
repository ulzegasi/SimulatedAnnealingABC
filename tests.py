"""
Simulated Annealing ABC (SABC) algorithm
Basic tests
"""

from pathlib import Path
from simulated_annealing_abc import (
    sabc,
    DifferentialEvolution,
    save_sabc_result,
    load_sabc_result,
)

# Create a dummy SABCResult object
from simulated_annealing_abc.sabc import SABCResult
import numpy as np  
population = np.random.rand(10, 2)  # 10 particles in 2D parameter space
u = np.random.rand(10, 3)  # 10 particles with 3 summary statistics
rho = np.random.rand(10, 3)  # 10 particles with 3 distances            
state = None  # For simplicity, we set state to None in this test
out = SABCResult(population=population, u=u, rho=rho, state=state)
# Path of *this* script
HERE = Path(__file__).resolve().parent
# Target file: ./test_results/test.pkl
out_path = HERE / "test_results" / "test.pkl"
save_sabc_result(out, out_path)