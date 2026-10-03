"""Iterative methods for symmetric positive-definite sparse systems."""
from dataclasses import dataclass
import numpy as np
from csr import CSRMatrix


@dataclass
class SolverResult:
    x: np.ndarray
    residuals: np.ndarray
    iterations: int
    converged: bool

    @property
    def relative_residual(self) -> float:
        if self.residuals[0] == 0:
            return 0.0
        return float(self.residuals[-1] / self.residuals[0])


def _result(x, residuals, tol):
    residuals = np.asarray(residuals, dtype=float)
    rel = 0.0 if residuals[0] == 0 else residuals[-1] / residuals[0]
    return SolverResult(x, residuals, len(residuals) - 1, bool(rel < tol))


def jacobi(A: CSRMatrix, b, x0=None, tol=1e-8, max_iter=10_000, relaxation=1.0):
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    diag = A.diagonal()
    r = b - A.matvec(x)
    residuals = [np.linalg.norm(r)]
    if residuals[0] == 0:
        return _result(x, residuals, tol)

    for _ in range(max_iter):
        x += relaxation * r / diag
        r = b - A.matvec(x)
        residuals.append(np.linalg.norm(r))
        if residuals[-1] < tol * residuals[0]:
            break
    return _result(x, residuals, tol)


def steepest_descent(A: CSRMatrix, b, x0=None, tol=1e-8, max_iter=10_000):
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    r = b - A.matvec(x)
    residuals = [np.linalg.norm(r)]
    if residuals[0] == 0:
        return _result(x, residuals, tol)

    for _ in range(max_iter):
        Ar = A.matvec(r)
        denominator = np.dot(r, Ar)
        if denominator == 0:
            break
        alpha = np.dot(r, r) / denominator
        x += alpha * r
        r -= alpha * Ar
        residuals.append(np.linalg.norm(r))
        if residuals[-1] < tol * residuals[0]:
            break
    return _result(x, residuals, tol)


def conjugate_gradient(A: CSRMatrix, b, x0=None, tol=1e-8, max_iter=10_000):
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    r = b - A.matvec(x)
    p = r.copy()
    rr = np.dot(r, r)
    residuals = [np.sqrt(rr)]
    if residuals[0] == 0:
        return _result(x, residuals, tol)

    for _ in range(max_iter):
        Ap = A.matvec(p)
        denominator = np.dot(p, Ap)
        if denominator == 0:
            break
        alpha = rr / denominator
        x += alpha * p
        r -= alpha * Ap
        rr_new = np.dot(r, r)
        residuals.append(np.sqrt(rr_new))
        if residuals[-1] < tol * residuals[0]:
            rr = rr_new
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    return _result(x, residuals, tol)


def apply_preconditioner(A: CSRMatrix, r, kind="SSOR", omega=1.77):
    """Apply Id, diagonal Jacobi, or the project SSOR preconditioner."""
    r = np.asarray(r, dtype=float)
    if kind == "Id":
        return r.copy()
    if kind == "D":
        return r / A.diagonal()
    if kind != "SSOR":
        raise ValueError("Preconditioner must be one of: Id, D, SSOR")

    z = r.copy()
    diag = A.diagonal()

    # Backward triangular solve: (D - omega E^T)^-1 r
    for i in range(A.n - 1, -1, -1):
        start, end = A.row_ptr[i], A.row_ptr[i + 1]
        for jj in range(start, end):
            col = A.col_ind[jj]
            if col > i:
                z[i] -= omega * A.values[jj] * z[col]
        z[i] /= diag[i]

    # Middle multiplication by D
    z *= diag

    # Forward triangular solve: (D - omega E)^-1
    for i in range(A.n):
        start, end = A.row_ptr[i], A.row_ptr[i + 1]
        for jj in range(start, end):
            col = A.col_ind[jj]
            if col < i:
                z[i] -= omega * A.values[jj] * z[col]
        z[i] /= diag[i]

    return z


def preconditioned_conjugate_gradient(
    A: CSRMatrix,
    b,
    x0=None,
    tol=1e-8,
    max_iter=10_000,
    preconditioner="SSOR",
    omega=1.77,
):
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    r = b - A.matvec(x)
    z = apply_preconditioner(A, r, preconditioner, omega)
    p = z.copy()
    rz = np.dot(r, z)
    residuals = [np.linalg.norm(r)]
    if residuals[0] == 0:
        return _result(x, residuals, tol)

    for _ in range(max_iter):
        Ap = A.matvec(p)
        denominator = np.dot(p, Ap)
        if denominator == 0:
            break
        alpha = rz / denominator
        x += alpha * p
        r -= alpha * Ap
        residuals.append(np.linalg.norm(r))
        if residuals[-1] < tol * residuals[0]:
            break
        z_new = apply_preconditioner(A, r, preconditioner, omega)
        rz_new = np.dot(r, z_new)
        beta = rz_new / rz
        p = z_new + beta * p
        z = z_new
        rz = rz_new
    return _result(x, residuals, tol)


SOLVERS = {
    "jacobi": jacobi,
    "gradient": steepest_descent,
    "cg": conjugate_gradient,
    "pcg": preconditioned_conjugate_gradient,
}
