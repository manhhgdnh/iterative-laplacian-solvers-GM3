"""Minimal CSR matrix utilities used by the numerical solvers."""
from dataclasses import dataclass
import numpy as np


@dataclass
class CSRMatrix:
    row_ptr: np.ndarray
    col_ind: np.ndarray
    values: np.ndarray
    n: int

    def __post_init__(self):
        self.row_ptr = np.asarray(self.row_ptr, dtype=int)
        self.col_ind = np.asarray(self.col_ind, dtype=int)
        self.values = np.asarray(self.values, dtype=float)
        if self.row_ptr.shape != (self.n + 1,):
            raise ValueError("row_ptr must contain n + 1 entries")
        if len(self.col_ind) != len(self.values):
            raise ValueError("col_ind and values must have the same length")

    @property
    def nnz(self) -> int:
        return len(self.values)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise ValueError(f"Expected vector of shape ({self.n},), got {x.shape}")
        result = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            start, end = self.row_ptr[i], self.row_ptr[i + 1]
            result[i] = np.dot(self.values[start:end], x[self.col_ind[start:end]])
        return result

    def diagonal(self) -> np.ndarray:
        diag = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            start, end = self.row_ptr[i], self.row_ptr[i + 1]
            cols = self.col_ind[start:end]
            vals = self.values[start:end]
            positions = np.flatnonzero(cols == i)
            if positions.size == 0:
                raise ValueError(f"Missing diagonal entry in row {i}")
            diag[i] = vals[positions[0]]
        return diag

    def to_dense(self) -> np.ndarray:
        dense = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            start, end = self.row_ptr[i], self.row_ptr[i + 1]
            dense[i, self.col_ind[start:end]] = self.values[start:end]
        return dense
