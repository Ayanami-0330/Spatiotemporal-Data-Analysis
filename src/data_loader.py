"""
Dataset I/O and basic preprocessing utilities.

This module provides a small set of helpers to

* read the raw array ``vector_64.npy`` with shape ``(nt, ny, nx, 2)``,
* form a time‑averaged velocity field,
* construct the fluctuation field by subtracting the mean profile,
* reshape the fluctuations into a 2‑D data matrix of size ``(N, T)``,
* and derive scalar diagnostics such as vorticity and divergence
  using finite differences under periodic boundary conditions.

The main entry point is :func:`load_and_preprocess`, which bundles all
of the above into a single dictionary of arrays.
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np


# ── dataset meta information ─────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetMeta:
    """Light container for basic grid / time information."""

    nx: int
    ny: int
    nt: int
    dt: float

    @property
    def n_dof(self) -> int:
        return self.nx * self.ny * 2

    @property
    def total_time(self) -> float:
        return self.nt * self.dt


# canonical meta object used throughout the project
META = DatasetMeta(nx=64, ny=64, nt=15000, dt=0.2)

# module‑level aliases retained for backward compatibility
NX, NY = META.nx, META.ny
NT = META.nt
DT = META.dt
N_DOF = META.n_dof
T_TOTAL = META.total_time

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw(fname: str = "vector_64.npy") -> np.ndarray:
    """Load the raw 4‑D velocity array with shape ``(nt, ny, nx, 2)``."""
    data = np.load(DATA_DIR / fname)
    assert data.shape == (NT, NY, NX, 2), f"Unexpected shape {data.shape}"
    print(
        f"[data] opened {fname}: "
        f"shape={data.shape}, dtype={data.dtype}, "
        f"size={data.nbytes / 1e9:.2f} GB"
    )
    return data


def compute_mean_field(data: np.ndarray) -> np.ndarray:
    """Return the temporal mean velocity field ``ū(x, y)``."""
    return data.mean(axis=0)


def compute_fluctuation(
    data: np.ndarray,
    mean_field: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the fluctuation field ``u' = u − ū``."""
    if mean_field is None:
        mean_field = compute_mean_field(data)
    return data - mean_field[np.newaxis, ...]


def build_data_matrix(field: np.ndarray) -> np.ndarray:
    """
    Stack all snapshots into a two‑dimensional data matrix of shape
    ``(N, T)``, where each column corresponds to one time instant.

    The ordering is ``[u_x (all grid points), u_y (all grid points)]``.
    """
    n_snapshots = field.shape[0]
    # (nt, ny, nx, 2) → (nt, 2, ny, nx) → (nt, 2*ny*nx)
    flat = field.transpose(0, 3, 1, 2).reshape(n_snapshots, -1)  # (T, N)
    return flat.T  # (N, T)


# ── derived scalar fields (periodic finite differences) ─────────────────
def _central_diff_periodic(
    arr: np.ndarray,
    axis: int,
    dx: float = 1.0,
) -> np.ndarray:
    """Second‑order central difference with periodic boundary conditions."""
    forward = np.roll(arr, -1, axis=axis)
    backward = np.roll(arr, 1, axis=axis)
    return (forward - backward) / (2.0 * dx)


def compute_vorticity(data: np.ndarray, L: float | None = None) -> np.ndarray:
    """
    Compute the vorticity field
    :math:`\\omega = \\partial_x u_y - \\partial_y u_x` with shape
    ``(nt, ny, nx)`` from a velocity array of shape ``(nt, ny, nx, 2)``.
    """
    dx = (L / NX) if L is not None else 1.0
    dy = (L / NY) if L is not None else 1.0
    ux = data[..., 0]  # (nt, ny, nx)
    uy = data[..., 1]
    d_uy_dx = _central_diff_periodic(uy, axis=2, dx=dx)
    d_ux_dy = _central_diff_periodic(ux, axis=1, dx=dy)
    return d_uy_dx - d_ux_dy


def compute_divergence(data: np.ndarray, L: float | None = None) -> np.ndarray:
    """
    Compute the divergence
    :math:`\\nabla\\cdot u = \\partial_x u_x + \\partial_y u_y`.
    """
    dx = (L / NX) if L is not None else 1.0
    dy = (L / NY) if L is not None else 1.0
    ux = data[..., 0]
    uy = data[..., 1]
    d_ux_dx = _central_diff_periodic(ux, axis=2, dx=dx)
    d_uy_dy = _central_diff_periodic(uy, axis=1, dx=dy)
    return d_ux_dx + d_uy_dy


# ── high‑level convenience wrapper ──────────────────────────────────────
def load_and_preprocess(fname: str = "vector_64.npy"):
    """
    Load the dataset and construct a dictionary with the most commonly
    used derived quantities for the later analysis stages.

    The returned mapping contains:

    * ``raw``        – original velocity snapshots, ``(nt, ny, nx, 2)``,
    * ``mean_field`` – time‑averaged velocity field, ``(ny, nx, 2)``,
    * ``fluctuation`` – deviation from the mean, ``(nt, ny, nx, 2)``,
    * ``data_matrix`` – flattened fluctuation matrix, ``(N, T)``,
    * ``vorticity``  – scalar vorticity field, ``(nt, ny, nx)``,
    * ``divergence`` – scalar divergence field, ``(nt, ny, nx)``.
    """
    raw = load_raw(fname)
    mean_velocity = compute_mean_field(raw)
    fluctuation = compute_fluctuation(raw, mean_velocity)
    data_matrix = build_data_matrix(fluctuation)
    vorticity = compute_vorticity(raw)
    divergence = compute_divergence(raw)
    print(f"[data] data_matrix: {data_matrix.shape}")
    print(f"[data] vorticity field: {vorticity.shape}")
    print(f"[data] divergence field: {divergence.shape}")
    return dict(
        raw=raw,
        mean_field=mean_velocity,
        fluctuation=fluctuation,
        data_matrix=data_matrix,
        vorticity=vorticity,
        divergence=divergence,
    )
