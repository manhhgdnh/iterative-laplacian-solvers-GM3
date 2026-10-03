"""Finite-difference discretizations of 1D, 2D and 3D Laplacian problems."""
import numpy as np
from csr import CSRMatrix


def laplacian_1d_csr(n: int, k: float = 10.0):
    """Discretize -u'' + k u = f on [0,1] with u(0)=1 and u(1)=0."""
    h = 1.0 / (n + 1.0)
    inv_h2 = 1.0 / h**2
    row_ptr, col_ind, values = [], [], []
    count = 0

    for i in range(n):
        row_ptr.append(count)
        if i > 0:
            col_ind.append(i - 1)
            values.append(-inv_h2)
            count += 1
        col_ind.append(i)
        values.append(2.0 * inv_h2 + k)
        count += 1
        if i + 1 < n:
            col_ind.append(i + 1)
            values.append(-inv_h2)
            count += 1
    row_ptr.append(count)

    rhs = np.zeros(n)
    rhs[0] = inv_h2
    return CSRMatrix(row_ptr, col_ind, values, n), rhs


def laplacian_2d_csr(n: int, k: float = 0.0):
    """Five-point finite-difference Laplacian on the unit square."""
    h = 1.0 / (n + 1.0)
    inv_h2 = 1.0 / h**2
    row_ptr, col_ind, values = [], [], []
    count = 0

    for j in range(n):
        for i in range(n):
            idx = j * n + i
            row_ptr.append(count)
            if j > 0:
                col_ind.append(idx - n)
                values.append(-inv_h2)
                count += 1
            if i > 0:
                col_ind.append(idx - 1)
                values.append(-inv_h2)
                count += 1
            col_ind.append(idx)
            values.append(4.0 * inv_h2 + k)
            count += 1
            if i + 1 < n:
                col_ind.append(idx + 1)
                values.append(-inv_h2)
                count += 1
            if j + 1 < n:
                col_ind.append(idx + n)
                values.append(-inv_h2)
                count += 1
    row_ptr.append(count)

    rhs = np.zeros(n * n)
    for j in range(n):
        y = (j + 1) * h
        for i in range(n):
            x = (i + 1) * h
            idx = j * n + i
            if abs(x - 0.5) < 0.2 and abs(y - 0.5) < 0.05:
                rhs[idx] = x - 0.5
    return CSRMatrix(row_ptr, col_ind, values, n * n), rhs


def laplacian_3d_csr(n: int, k: float = 0.0):
    """Seven-point finite-difference Laplacian on the unit cube."""
    h = 1.0 / (n + 1.0)
    inv_h2 = 1.0 / h**2
    row_ptr, col_ind, values = [], [], []
    count = 0

    for z_idx in range(n):
        for y_idx in range(n):
            for x_idx in range(n):
                idx = z_idx * n * n + y_idx * n + x_idx
                row_ptr.append(count)
                neighbors = (
                    (z_idx > 0, idx - n * n),
                    (y_idx > 0, idx - n),
                    (x_idx > 0, idx - 1),
                )
                for condition, col in neighbors:
                    if condition:
                        col_ind.append(col)
                        values.append(-inv_h2)
                        count += 1

                col_ind.append(idx)
                values.append(6.0 * inv_h2 + k)
                count += 1

                neighbors = (
                    (x_idx + 1 < n, idx + 1),
                    (y_idx + 1 < n, idx + n),
                    (z_idx + 1 < n, idx + n * n),
                )
                for condition, col in neighbors:
                    if condition:
                        col_ind.append(col)
                        values.append(-inv_h2)
                        count += 1
    row_ptr.append(count)

    rhs = np.zeros(n**3)
    for z_idx in range(n):
        z = (z_idx + 1) * h
        for y_idx in range(n):
            y = (y_idx + 1) * h
            for x_idx in range(n):
                x = (x_idx + 1) * h
                idx = z_idx * n * n + y_idx * n + x_idx
                rhs[idx] = 2.0 * (z - 0.5) * np.exp(
                    -100.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2)
                )
    return CSRMatrix(row_ptr, col_ind, values, n**3), rhs
