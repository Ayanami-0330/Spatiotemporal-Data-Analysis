"""
Author: Tan Kailai (谭开来, e1554333@u.nus.edu)
Date: 2026-03-01
Course: NUS ME5311 Project 1
"""

"""
High-level driver script for ME5311 Project 1.

This file only coordinates the analysis workflow:
    1. load and preprocess the raw vector field snapshots,
    2. perform a low‑dimensional modal decomposition (SVD/PCA),
    3. analyse spatial and temporal spectra,
    4. examine symmetry and anisotropy indicators.

All figures are written to the ``figures/`` directory so they can be
directly used when preparing the report.
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from src import data_loader as dl
from src import svd_analysis as svd
from src import spectral_analysis as spectral
from src import symmetry_analysis as symmetry
from src import visualization as viz


def _step0_preprocessing():
    """Load the raw snapshots and construct all basic diagnostic fields."""
    print("\n[Step 0] data loading and basic diagnostics")

    data_pack = dl.load_and_preprocess()
    raw_snapshots   = data_pack["raw"]           # (15000, 64, 64, 2)
    mean_profile    = data_pack["mean_field"]    # (64, 64, 2)
    fluct_field     = data_pack["fluctuation"]   # (15000, 64, 64, 2)
    vorticity_field = data_pack["vorticity"]     # (15000, 64, 64)
    divergence_field = data_pack["divergence"]   # (15000, 64, 64)

    # Quick visual impression: mean flow, one instantaneous snapshot,
    # and the corresponding vorticity/divergence.
    viz.plot_vector_snapshot(
        mean_profile,
        title="Time‑averaged velocity field",
        save_name="step0_mean_field",
    )
    viz.plot_vector_snapshot(
        raw_snapshots[0],
        title="Instantaneous snapshot at t = 0",
        save_name="step0_snapshot_t0",
    )
    viz.plot_scalar_field(
        vorticity_field[0],
        title="Vorticity $\\omega$ at t = 0",
        save_name="step0_vorticity_t0",
    )
    viz.plot_scalar_field(
        divergence_field[0],
        title="Divergence $\\nabla\\cdot u$ at t = 0",
        save_name="step0_divergence_t0",
    )
    # Global kinetic energy evolution and basic PDFs of vorticity/divergence.
    viz.plot_kinetic_energy_timeseries(
        raw_snapshots,
        dt=dl.DT,
        save_name="step0_kinetic_energy_timeseries",
    )
    viz.plot_scalar_histogram(
        vorticity_field,
        bins=80,
        title="PDF of vorticity over space–time",
        xlabel="Vorticity $\\omega$",
        save_name="step0_vorticity_pdf",
    )
    viz.plot_scalar_histogram(
        divergence_field,
        bins=80,
        title="PDF of divergence over space–time",
        xlabel="Divergence $\\nabla\\cdot u$",
        save_name="step0_divergence_pdf",
    )
    return data_pack


def _step1_svd_decomposition(snapshot_matrix: np.ndarray):
    """Compute SVD of the fluctuation field and store basic statistics."""
    print("\n[Step 1] SVD on fluctuation field")
    return svd.run(snapshot_matrix, dt=dl.DT, ny=dl.NY, nx=dl.NX, n_modes=6)


def _step2_spectral_analysis(fluct_field: np.ndarray,
                             vorticity_field: np.ndarray):
    """Run both spatial and temporal spectral analysis."""
    print("\n[Step 2] spectral characterisation")

    # Spectra based on the velocity fluctuation.
    spectra = spectral.run(fluct_field, dt=dl.DT)

    # Additional check: spatial spectrum of vorticity.
    print("\n  [extra] radial spectrum of vorticity")
    vor_fft = np.fft.fft2(vorticity_field, axes=(1, 2))
    psd_vor_2d = np.mean(np.abs(vor_fft) ** 2, axis=0) / (dl.NX * dl.NY)
    k_bins, radial_vor = spectral.radial_spectrum(psd_vor_2d)
    viz.plot_radial_spectrum(
        k_bins,
        radial_vor,
        title="Radial PSD — vorticity",
        save_name="spectral_radial_vorticity",
    )
    spectral.detect_peak_wavenumbers(k_bins, radial_vor)
    spectra["psd_vorticity_2d"] = psd_vor_2d
    spectra["radial_vorticity"] = radial_vor
    return spectra


def _step3_symmetry_and_anisotropy(fluct_field: np.ndarray,
                                   spectra: dict,
                                   svd_out: dict):
    """Use spectra and SVD modes to assess symmetry/anisotropy."""
    print("\n[Step 3] symmetry and anisotropy indicators")

    leading_modes = svd.extract_spatial_modes(svd_out["U"], n_modes=6)
    sym_summary = symmetry.run(
        fluct_field,
        spectra["psd_total_2d"],
        svd_modes=leading_modes,
    )

    # Direct comparison between ux and uy radial spectra (direction‑wise view).
    component_spectra = spectral.compare_components_spatial(fluct_field)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(
        component_spectra["k_bins"],
        component_spectra["psd_ux_radial"],
        "o-",
        ms=3,
        color="tab:blue",
        label="$u_x$",
    )
    ax.semilogy(
        component_spectra["k_bins"],
        component_spectra["psd_uy_radial"],
        "s-",
        ms=3,
        color="tab:orange",
        label="$u_y$",
    )
    ax.set_xlabel("Radial wave number $k$")
    ax.set_ylabel("Spectral density $E(k)$")
    ax.set_title("Direction‑resolved radial spectra: $u_x$ versus $u_y$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    viz.savefig(fig, "spectral_component_comparison")

    return dict(symmetry=sym_summary, components=component_spectra)


def main():
    """Run the complete analysis pipeline end‑to‑end."""
    print("=" * 60)
    print("  ME5311 Project 1 - Spatio-temporal data exploration")
    print("=" * 60)

    bundle = _step0_preprocessing()
    svd_results = _step1_svd_decomposition(bundle["data_matrix"])
    spectral_results = _step2_spectral_analysis(
        bundle["fluctuation"],
        bundle["vorticity"],
    )
    symmetry_results = _step3_symmetry_and_anisotropy(
        bundle["fluctuation"],
        spectral_results,
        svd_results,
    )

    print("\n" + "=" * 60)
    print("  Analysis complete. Figures are stored under:", viz.FIG_DIR)
    print("=" * 60)

    return dict(
        bundle=bundle,
        svd=svd_results,
        spectral=spectral_results,
        symmetry=symmetry_results["symmetry"],
        anisotropy=symmetry_results["components"],
    )


if __name__ == "__main__":
    results = main()
