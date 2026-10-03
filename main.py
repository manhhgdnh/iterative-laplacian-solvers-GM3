#!/usr/bin/env python3
"""Command-line entry point for the Laplacian solver project."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
from discretization import laplacian_1d_csr, laplacian_2d_csr, laplacian_3d_csr
from solvers import SOLVERS
from visualization import save_1d_solution, save_2d_field, save_2d_potential, write_vector_vtk


def parse_args():
    parser = argparse.ArgumentParser(description="Solve finite-difference Laplacian systems with iterative methods.")
    parser.add_argument("--dim", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--method", choices=tuple(SOLVERS), default="pcg")
    parser.add_argument("--n", type=int, default=None, help="Interior grid points per spatial direction")
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=10_000)
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = {1: (100, 10.0), 2: (50, 0.0), 3: (20, 0.0)}
    default_n, default_k = defaults[args.dim]
    n = args.n if args.n is not None else default_n
    k = args.k if args.k is not None else default_k

    generators = {1: laplacian_1d_csr, 2: laplacian_2d_csr, 3: laplacian_3d_csr}
    A, b = generators[args.dim](n, k)
    result = SOLVERS[args.method](A, b, tol=args.tol, max_iter=args.max_iter)

    print(f"Dimension: {args.dim}D")
    print(f"Unknowns: {A.n}")
    print(f"Non-zero coefficients: {A.nnz}")
    print(f"Method: {args.method.upper()}")
    print(f"Iterations: {result.iterations}")
    print(f"Relative residual: {result.relative_residual:.3e}")
    print(f"Converged: {result.converged}")

    figures = ROOT / "results" / "figures"
    if args.dim == 1:
        save_1d_solution(result.x, figures / "solution_1d.png")
    elif args.dim == 2:
        save_2d_potential(result.x, n, figures / "potential_2d.png")
        save_2d_field(result.x, n, figures / "field_2d_generated.png")
    else:
        potential = result.x.reshape((n, n, n))
        h = 1.0 / (n + 1)
        gx, gy, gz = np.gradient(potential, h, h, h)
        write_vector_vtk(ROOT / "results" / "vtk" / "field_3d.vtk", gx, gy, gz, h)
        print("VTK field written to results/vtk/field_3d.vtk")


if __name__ == "__main__":
    main()
