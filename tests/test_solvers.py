import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discretization import laplacian_1d_csr, laplacian_2d_csr
from solvers import jacobi, steepest_descent, conjugate_gradient, preconditioned_conjugate_gradient


class LaplacianSolverTests(unittest.TestCase):
    def test_csr_matrix_is_symmetric_positive_definite(self):
        A, _ = laplacian_2d_csr(4, 0.0)
        dense = A.to_dense()
        self.assertTrue(np.allclose(dense, dense.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(dense) > 0))

    def test_iterative_solvers_match_dense_reference(self):
        A, b = laplacian_1d_csr(20, 10.0)
        reference = np.linalg.solve(A.to_dense(), b)
        solvers = (
            lambda: jacobi(A, b, tol=1e-8, max_iter=20_000),
            lambda: steepest_descent(A, b, tol=1e-8, max_iter=20_000),
            lambda: conjugate_gradient(A, b, tol=1e-10, max_iter=1_000),
            lambda: preconditioned_conjugate_gradient(A, b, tol=1e-8, max_iter=1_000),
        )
        for solve in solvers:
            result = solve()
            self.assertLess(np.linalg.norm(result.x - reference) / np.linalg.norm(reference), 1e-6)

    def test_pcg_converges_on_2d_problem(self):
        A, b = laplacian_2d_csr(12, 0.0)
        result = preconditioned_conjugate_gradient(A, b, tol=1e-8, max_iter=2_000)
        residual = np.linalg.norm(A.matvec(result.x) - b) / np.linalg.norm(b)
        self.assertLess(residual, 1e-7)


if __name__ == "__main__":
    unittest.main()
