"""
Fourier‑based characterisation of spatial and temporal scales.

The routines in this file focus on summarising the fluctuation field
in frequency / wavenumber space. In particular, we construct

* two‑dimensional power spectra in wavenumber space,
* azimuthally averaged (radial) spectra,
* spatially averaged temporal PSDs,
* and a few lightweight helpers for identifying energetic peaks.
"""

import numpy as np
from . import data_loader as dl
from . import visualization as viz


# ── 空间谱分析 ───────────────────────────────────────────────
def spatial_psd_2d(data: np.ndarray, component: int = 0) -> np.ndarray:
    """
    Compute a time‑averaged two‑dimensional power spectrum.

    Parameters
    ----------
    data : ndarray
        Velocity snapshots with shape ``(nt, ny, nx, 2)``.
    component : int
        Which component to analyse, ``0`` for :math:`u_x`,
        ``1`` for :math:`u_y`.
    """
    field = data[..., component]            # (nt, ny, nx)
    fhat = np.fft.fft2(field, axes=(1, 2))  # (nt, ny, nx)
    psd = np.mean(np.abs(fhat) ** 2, axis=0) / (dl.NX * dl.NY)
    return psd


def radial_spectrum(psd_2d: np.ndarray):
    """
    Reduce a 2‑D spectrum to an isotropically averaged 1‑D profile.

    Returns integer wavenumber bins and the corresponding radial energy.
    """
    ny, nx = psd_2d.shape
    # Wavenumber grid.
    kx = np.fft.fftfreq(nx, d=1.0) * nx
    ky = np.fft.fftfreq(ny, d=1.0) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_max = int(np.floor(K.max()))
    k_bins = np.arange(0, k_max + 1)
    psd_radial = np.zeros(len(k_bins))
    for i, k in enumerate(k_bins):
        shell = (K >= k - 0.5) & (K < k + 0.5)
        if shell.any():
            psd_radial[i] = psd_2d[shell].sum()
    return k_bins, psd_radial


def detect_peak_wavenumbers(
    k_bins: np.ndarray,
    psd_radial: np.ndarray,
    n_peaks: int = 5,
):
    """
    Identify the most energetic discrete wavenumbers in a radial spectrum.

    The zero mode (DC component) is explicitly ignored.
    """
    psd_copy = psd_radial.copy()
    psd_copy[0] = 0  # ignore DC level
    idx = np.argsort(psd_copy)[::-1][:n_peaks]
    peaks = [(k_bins[i], psd_radial[i]) for i in idx]
    print("[spectral] dominant wavenumbers (radial PSD):")
    for k, p in peaks:
        print(f"    k = {k:.0f},  E(k) = {p:.4e}")
    return peaks


# ── 时间谱分析 ───────────────────────────────────────────────
def temporal_psd_avg(
    data: np.ndarray,
    component: int = 0,
    dt: float = dl.DT,
):
    """
    Compute a spatially averaged temporal power spectrum.

    For each grid point, a 1‑D FFT is taken along the time axis and the
    resulting PSDs are averaged over space.
    """
    field = data[..., component]  # (nt, ny, nx)
    nt = field.shape[0]
    fhat = np.fft.rfft(field, axis=0)  # (nt//2+1, ny, nx)
    psd = np.mean(np.abs(fhat) ** 2, axis=(1, 2)) / nt
    freqs = np.fft.rfftfreq(nt, d=dt)
    return freqs, psd


def detect_peak_frequencies(
    freqs: np.ndarray,
    psd: np.ndarray,
    n_peaks: int = 5,
):
    """Locate a few dominant temporal frequencies in a PSD curve."""
    psd_copy = psd.copy()
    psd_copy[0] = 0
    idx = np.argsort(psd_copy)[::-1][:n_peaks]
    peaks = [(freqs[i], psd[i]) for i in idx]
    print("[spectral] peak frequencies:")
    for f_val, p_val in peaks:
        print(f"    f = {f_val:.6f},  PSD = {p_val:.4e}")
    return peaks


# ── 分量对比 ─────────────────────────────────────────────────
def compare_components_spatial(data: np.ndarray):
    """
    Compare spatial spectra of :math:`u_x` and :math:`u_y`.

    This is used later as a simple anisotropy indicator.
    """
    psd_ux = spatial_psd_2d(data, component=0)
    psd_uy = spatial_psd_2d(data, component=1)
    k_bins, rad_ux = radial_spectrum(psd_ux)
    _,      rad_uy = radial_spectrum(psd_uy)
    return dict(
        k_bins=k_bins,
        psd_ux_radial=rad_ux,
        psd_uy_radial=rad_uy,
        psd_ux_2d=psd_ux,
        psd_uy_2d=psd_uy,
    )


# ── 顶层运行函数 ────────────────────────────────────────────
def spectral_overview(data: np.ndarray, dt: float = dl.DT):
    """
    Compute a compact set of spatial and temporal spectra for the
    fluctuation field.

    Parameters
    ----------
    data : ndarray
        Fluctuation field with shape ``(nt, ny, nx, 2)``.
    dt : float
        Time step between successive snapshots.
    """
    results: dict[str, np.ndarray] = {}

    # ---- spatial spectra for ux ----
    psd_ux_2d = spatial_psd_2d(data, component=0)
    k_bins, rad_ux = radial_spectrum(psd_ux_2d)
    viz.plot_2d_spectrum(
        psd_ux_2d,
        title="2D wavenumber–energy map — $u_x$",
        save_name="spectral_2d_ux",
    )
    viz.plot_radial_spectrum(
        k_bins,
        rad_ux,
        title="Radial spectral density — $u_x$",
        save_name="spectral_radial_ux",
    )

    # ---- spatial spectra for uy ----
    psd_uy_2d = spatial_psd_2d(data, component=1)
    _, rad_uy = radial_spectrum(psd_uy_2d)
    viz.plot_2d_spectrum(
        psd_uy_2d,
        title="2D wavenumber–energy map — $u_y$",
        save_name="spectral_2d_uy",
    )
    viz.plot_radial_spectrum(
        k_bins,
        rad_uy,
        title="Radial spectral density — $u_y$",
        save_name="spectral_radial_uy",
    )

    # ---- combined radial spectrum ----
    psd_total_2d = psd_ux_2d + psd_uy_2d
    _, rad_total = radial_spectrum(psd_total_2d)
    fig_rad, _ = viz.plot_radial_spectrum(
        k_bins,
        rad_total,
        title="Radial spectral density — combined components",
        save_name="spectral_radial_total",
    )

    # Additional derived views: compensated and cumulative spectra.
    viz.plot_compensated_spectrum(
        k_bins,
        rad_total,
        exponent=2.0,
        title="Compensated radial spectrum $k^2 E(k)$ — combined components",
        save_name="spectral_radial_total_compensated",
    )
    viz.plot_cumulative_spectrum(
        k_bins,
        rad_total,
        save_name="spectral_radial_total_cumulative",
    )

    # ---- spatial peaks ----
    peaks_spatial = detect_peak_wavenumbers(k_bins, rad_total)

    # ---- temporal spectra ----
    freqs_ux, tpsd_ux = temporal_psd_avg(data, component=0, dt=dt)
    freqs_uy, tpsd_uy = temporal_psd_avg(data, component=1, dt=dt)
    viz.plot_temporal_psd(
        freqs_ux,
        tpsd_ux,
        title="Temporal PSD (spatially averaged) — $u_x$",
        save_name="spectral_temporal_ux",
    )
    viz.plot_temporal_psd(
        freqs_uy,
        tpsd_uy,
        title="Temporal PSD (spatially averaged) — $u_y$",
        save_name="spectral_temporal_uy",
    )
    peaks_temporal = detect_peak_frequencies(freqs_ux, tpsd_ux)

    results.update(
        psd_ux_2d=psd_ux_2d,
        psd_uy_2d=psd_uy_2d,
        psd_total_2d=psd_total_2d,
        k_bins=k_bins,
        rad_ux=rad_ux,
        rad_uy=rad_uy,
        rad_total=rad_total,
        peaks_spatial=peaks_spatial,
        freqs_ux=freqs_ux,
        tpsd_ux=tpsd_ux,
        freqs_uy=freqs_uy,
        tpsd_uy=tpsd_uy,
        peaks_temporal=peaks_temporal,
    )
    return results


def run(data: np.ndarray, dt: float = dl.DT):
    """
    Backwards‑compatible wrapper which forwards to :func:`spectral_overview`.
    """
    return spectral_overview(data=data, dt=dt)
