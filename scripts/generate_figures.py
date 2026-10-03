#!/usr/bin/env python3
"""Generate reproducible figures used by the README."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discretization import laplacian_1d_csr, laplacian_2d_csr
from solvers import jacobi, steepest_descent, conjugate_gradient, preconditioned_conjugate_gradient
from visualization import save_1d_solution, save_convergence, save_2d_potential, save_2d_field

OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# 1D solution and convergence (same grid size used in the report).
A1, b1 = laplacian_1d_csr(100, 10.0)
results_1d = {
    "Jacobi": jacobi(A1, b1, max_iter=10_000),
    "Gradient": steepest_descent(A1, b1, max_iter=10_000),
    "CG": conjugate_gradient(A1, b1, max_iter=10_000),
    "PCG (SSOR)": preconditioned_conjugate_gradient(A1, b1, max_iter=10_000),
}
save_1d_solution(results_1d["PCG (SSOR)"].x, OUT / "solution_1d.png")
save_convergence(results_1d, OUT / "convergence_1d.png", "Convergence comparison - 1D")

# 2D potential and field.
A2, b2 = laplacian_2d_csr(50, 0.0)
r2 = preconditioned_conjugate_gradient(A2, b2, max_iter=10_000)
save_2d_potential(r2.x, 50, OUT / "potential_2d.png")
save_2d_field(r2.x, 50, OUT / "field_2d_generated.png")

# Sparse structure of a small representative 2D Laplacian.
A_small, _ = laplacian_2d_csr(12, 0.0)
fig, ax = plt.subplots(figsize=(6, 6))
ax.spy(A_small.to_dense(), markersize=1.5)
ax.set_title("Sparsity pattern - 2D five-point Laplacian")
ax.set_xlabel("Column")
ax.set_ylabel("Row")
fig.tight_layout()
fig.savefig(OUT / "sparsity_pattern_2d.png", dpi=180)
plt.close(fig)

# Iteration counts reported in the academic report.
methods = ["Jacobi", "Gradient", "CG", "PCG"]
counts = {
    "1D": [10000, 10000, 101, 15],
    "2D": [3597, 3599, 107, 31],
    "3D": [1630, 1611, 74, 24],
}
x = np.arange(len(methods))
width = 0.24
fig, ax = plt.subplots(figsize=(8, 5))
for offset, (label, values) in zip((-width, 0, width), counts.items()):
    ax.bar(x + offset, values, width, label=label)
ax.set_yscale("log")
ax.set_xticks(x, methods)
ax.set_ylabel("Iterations (log scale)")
ax.set_title("Iteration counts reported in the project")
ax.legend()
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "reported_iteration_counts.png", dpi=180)
plt.close(fig)
