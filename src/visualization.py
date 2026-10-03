"""Visualization and VTK export helpers."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_1d_solution(solution, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(solution)
    h = 1.0 / (n + 1)
    x = np.linspace(0.0, 1.0, n + 2)
    u = np.zeros(n + 2)
    u[0] = 1.0
    u[1:-1] = solution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, u, linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("1D finite-difference solution")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_convergence(results, output_path, title="Relative residual convergence"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, result in results.items():
        rel = result.residuals / result.residuals[0]
        ax.plot(np.arange(len(rel)), np.log10(rel), label=name)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Iteration (symlog scale)")
    ax.set_ylabel(r"$\log_{10}(\|r_k\|/\|r_0\|)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_2d_potential(solution, n, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    potential = np.asarray(solution).reshape((n, n))
    h = 1.0 / (n + 1)
    extent = [h, 1.0 - h, h, 1.0 - h]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(potential, origin="lower", extent=extent, aspect="equal")
    fig.colorbar(im, ax=ax, label="Potential")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D numerical potential")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_2d_field(solution, n, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    potential = np.asarray(solution).reshape((n, n))
    h = 1.0 / (n + 1)
    coords = (np.arange(n) + 1) * h
    xx, yy = np.meshgrid(coords, coords)
    gy, gx = np.gradient(potential, h, h)
    norm = np.hypot(gx, gy)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.streamplot(xx, yy, gx, gy, density=1.5, linewidth=0.8)
    step = max(1, n // 20)
    ax.quiver(xx[::step, ::step], yy[::step, ::step], gx[::step, ::step], gy[::step, ::step], alpha=0.35)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D field reconstructed from the potential")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_vector_vtk(path, ux, uy, uz, spacing):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = ux.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Laplacian vector field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {spacing} {spacing} {spacing}\n")
        f.write(f"SPACING {spacing} {spacing} {spacing}\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write("VECTORS field float\n")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{ux[i,j,k]} {uy[i,j,k]} {uz[i,j,k]}\n")
