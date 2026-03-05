"""
Compact SVD/PCA utilities applied to the fluctuation field.

Each snapshot of the velocity field is interpreted as a single point in
a very high‑dimensional state space. By performing a truncated SVD of
the data matrix we recover

* the distribution of energy across modes (singular spectrum),
* dominant spatial patterns (left singular vectors),
* and associated temporal coefficients and their spectra.
"""

import numpy as np
from . import data_loader as dl
from . import visualization as viz


def compute_compact_svd(data_matrix: np.ndarray, full_matrices: bool = False):
    """
    Apply an economy‑size SVD to a data matrix of shape ``(N, T)``.

    Returns
    -------
    U : ndarray, shape (N, K)
    sigma : ndarray, shape (K,)
    Vt : ndarray, shape (K, T)
        with ``K = min(N, T)``.
    """
    print("[svd] start compact SVD factorisation")
    U, sigma, Vt = np.linalg.svd(data_matrix, full_matrices=full_matrices)
    print(f"[svd] done: U={U.shape}, S={sigma.shape}, Vt={Vt.shape}")
    return U, sigma, Vt


def modal_energy_summary(sigma: np.ndarray):
    """
    Convert singular values into an energy spectrum and cumulative sum.

    Returns
    -------
    energy : ndarray
        Mode energy :math:`\\sigma_k^2`.
    cum_energy : ndarray
        Cumulative fraction of total energy.
    n_95, n_99 : int
        Number of leading modes required to capture 95% / 99% energy.
    """
    energy = sigma**2
    cum_energy = np.cumsum(energy) / energy.sum()
    n_95 = int(np.searchsorted(cum_energy, 0.95)) + 1
    n_99 = int(np.searchsorted(cum_energy, 0.99)) + 1
    print(f"[svd] modes for 95% energy: {n_95}, 99% energy: {n_99}")
    return energy, cum_energy, n_95, n_99


def extract_spatial_modes(
    U: np.ndarray,
    ny: int = dl.NY,
    nx: int = dl.NX,
    n_modes: int = 6,
):
    """
    Reshape the first ``n_modes`` columns of ``U`` into 2‑D fields.

    Returns a list of ``(ux_mode, uy_mode)`` arrays, each of shape
    ``(ny, nx)``.
    """
    half = ny * nx
    spatial_modes = []
    for i in range(n_modes):
        ux_mode = U[:half, i].reshape(ny, nx)
        uy_mode = U[half:, i].reshape(ny, nx)
        spatial_modes.append((ux_mode, uy_mode))
    return spatial_modes


def temporal_coefficients(
    sigma: np.ndarray,
    Vt: np.ndarray,
    n_modes: int = 6,
):
    """Return ``a_k(t) = σ_k v_k(t)`` for the first ``n_modes``."""
    return [sigma[i] * Vt[i, :] for i in range(n_modes)]


def temporal_coefficient_psd(
    sigma: np.ndarray,
    Vt: np.ndarray,
    dt: float = dl.DT,
    n_modes: int = 6,
):
    """
    Compute one‑sided power spectral densities of SVD time coefficients.

    Returns frequency grid and a list of PSD arrays.
    """
    nt = Vt.shape[1]
    freqs = np.fft.rfftfreq(nt, d=dt)
    psds = []
    for i in range(n_modes):
        coeff = sigma[i] * Vt[i, :]
        fft_coeff = np.fft.rfft(coeff)
        psd = (np.abs(fft_coeff) ** 2) / nt
        psds.append(psd)
    return freqs, psds


# ── 顶层运行函数 ────────────────────────────────────────────
def svd_workflow(
    data_matrix: np.ndarray,
    dt: float = dl.DT,
    ny: int = dl.NY,
    nx: int = dl.NX,
    n_modes: int = 6,
):
    """
    High‑level SVD analysis workflow.

    Performs decomposition, energy accounting, and basic visualisation
    of spatial modes and temporal behaviour.
    """
    U, sigma, Vt = compute_compact_svd(data_matrix)

    # Energy distribution across modes.
    energy, cum, n95, n99 = modal_energy_summary(sigma)
    viz.plot_singular_values(sigma, n_show=min(100, len(sigma)))
    viz.plot_svd_reconstruction_error(
        sigma,
        n_show=min(100, len(sigma)),
    )
    viz.plot_mode_energy_bars(sigma, n_modes=min(10, len(sigma)))

    # Leading spatial patterns.
    viz.plot_spatial_modes(U, ny, nx, n_modes=n_modes)

    # Time‑domain view of modal amplitudes.
    viz.plot_temporal_coefficients(Vt, sigma, dt, n_modes=n_modes)

    # Frequency‑domain view of modal amplitudes.
    freqs, psds = temporal_coefficient_psd(
        sigma,
        Vt,
        dt,
        n_modes=n_modes,
    )
    viz.plot_mode_temporal_psd(freqs, psds, n_modes=n_modes)

    return dict(
        U=U,
        sigma=sigma,
        Vt=Vt,
        energy=energy,
        cum_energy=cum,
        n95=n95,
        n99=n99,
        freqs=freqs,
        psds=psds,
    )


def run(
    data_matrix: np.ndarray,
    dt: float = dl.DT,
    ny: int = dl.NY,
    nx: int = dl.NX,
    n_modes: int = 6,
):
    """
    Backwards‑compatible wrapper for the SVD analysis.

    This thin wrapper simply forwards to :func:`svd_workflow`. It is kept
    to avoid breaking older notebooks/scripts that still import
    ``svd_analysis.run``.
    """
    return svd_workflow(
        data_matrix=data_matrix,
        dt=dt,
        ny=ny,
        nx=nx,
        n_modes=n_modes,
    )
