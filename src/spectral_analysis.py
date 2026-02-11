"""
spectral_analysis.py — 傅里叶 / 功率谱分析
==========================================
- 2D 空间 FFT + 时间平均 PSD
- 径向功率谱（1D 化）
- 时间频率 PSD（空间平均）
- 峰值波数检测（外部强迫推断）
"""

import numpy as np
from . import data_loader as dl
from . import visualization as viz


# ── 空间谱分析 ───────────────────────────────────────────────
def spatial_psd_2d(data: np.ndarray, component: int = 0) -> np.ndarray:
    """
    对所有时间快照计算 2D 空间 FFT，然后时间平均功率谱。
    data: (nt, ny, nx, 2)
    component: 0=ux, 1=uy
    返回: psd_2d (ny, nx)
    """
    field = data[..., component]            # (nt, ny, nx)
    fhat = np.fft.fft2(field, axes=(1, 2))  # (nt, ny, nx)
    psd = np.mean(np.abs(fhat) ** 2, axis=0) / (dl.NX * dl.NY)
    return psd


def radial_spectrum(psd_2d: np.ndarray):
    """
    将 2D PSD 转化为径向（1D）功率谱。
    返回 (k_bins, psd_radial)。
    """
    ny, nx = psd_2d.shape
    # 波数网格
    kx = np.fft.fftfreq(nx, d=1.0) * nx   # 整数波数
    ky = np.fft.fftfreq(ny, d=1.0) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_max = int(np.floor(K.max()))
    k_bins = np.arange(0, k_max + 1)
    psd_radial = np.zeros(len(k_bins))
    for i, k in enumerate(k_bins):
        mask = (K >= k - 0.5) & (K < k + 0.5)
        if mask.any():
            psd_radial[i] = psd_2d[mask].sum()
    return k_bins, psd_radial


def detect_peak_wavenumbers(k_bins: np.ndarray, psd_radial: np.ndarray,
                            n_peaks: int = 5):
    """
    找出径向谱中最显著的 n_peaks 个波数峰值（排除 k=0 直流分量）。
    返回 [(k, psd_value), ...] 按能量降序。
    """
    psd_copy = psd_radial.copy()
    psd_copy[0] = 0  # 排除 DC
    idx = np.argsort(psd_copy)[::-1][:n_peaks]
    peaks = [(k_bins[i], psd_radial[i]) for i in idx]
    print("[spectral] Peak wavenumbers:")
    for k, p in peaks:
        print(f"    k = {k:.0f},  PSD = {p:.4e}")
    return peaks


# ── 时间谱分析 ───────────────────────────────────────────────
def temporal_psd_avg(data: np.ndarray, component: int = 0,
                     dt: float = dl.DT):
    """
    对每个网格点做时间方向 FFT，再空间平均，得到空间平均的时间 PSD。
    data: (nt, ny, nx, 2)
    返回 (freqs, psd_avg)
    """
    field = data[..., component]             # (nt, ny, nx)
    nt = field.shape[0]
    fhat = np.fft.rfft(field, axis=0)        # (nt//2+1, ny, nx)
    psd = np.mean(np.abs(fhat) ** 2, axis=(1, 2)) / nt
    freqs = np.fft.rfftfreq(nt, d=dt)
    return freqs, psd


def detect_peak_frequencies(freqs: np.ndarray, psd: np.ndarray,
                            n_peaks: int = 5):
    """找出时间 PSD 中最显著的 n_peaks 个频率峰值（排除 f=0）。"""
    psd_copy = psd.copy()
    psd_copy[0] = 0
    idx = np.argsort(psd_copy)[::-1][:n_peaks]
    peaks = [(freqs[i], psd[i]) for i in idx]
    print("[spectral] Peak frequencies:")
    for f, p in peaks:
        print(f"    f = {f:.6f},  PSD = {p:.4e}")
    return peaks


# ── 分量对比 ─────────────────────────────────────────────────
def compare_components_spatial(data: np.ndarray):
    """
    分别对 ux, uy 计算径向谱，用于各向异性诊断。
    返回 dict(k_bins, psd_ux, psd_uy)
    """
    psd_ux = spatial_psd_2d(data, component=0)
    psd_uy = spatial_psd_2d(data, component=1)
    k_bins, rad_ux = radial_spectrum(psd_ux)
    _,      rad_uy = radial_spectrum(psd_uy)
    return dict(k_bins=k_bins, psd_ux_radial=rad_ux, psd_uy_radial=rad_uy,
                psd_ux_2d=psd_ux, psd_uy_2d=psd_uy)


# ── 顶层运行函数 ────────────────────────────────────────────
def run(data: np.ndarray, dt: float = dl.DT):
    """
    完整空间 + 时间谱分析流水线。
    data: (nt, ny, nx, 2)
    """
    results = {}

    # ---- 空间谱：ux ----
    psd_ux_2d = spatial_psd_2d(data, component=0)
    k_bins, rad_ux = radial_spectrum(psd_ux_2d)
    viz.plot_2d_spectrum(psd_ux_2d, title="2D PSD — $u_x$",
                         save_name="spectral_2d_ux")
    viz.plot_radial_spectrum(k_bins, rad_ux,
                            title="Radial PSD — $u_x$",
                            save_name="spectral_radial_ux")

    # ---- 空间谱：uy ----
    psd_uy_2d = spatial_psd_2d(data, component=1)
    _, rad_uy = radial_spectrum(psd_uy_2d)
    viz.plot_2d_spectrum(psd_uy_2d, title="2D PSD — $u_y$",
                         save_name="spectral_2d_uy")
    viz.plot_radial_spectrum(k_bins, rad_uy,
                            title="Radial PSD — $u_y$",
                            save_name="spectral_radial_uy")

    # ---- 合成径向谱 ----
    psd_total_2d = psd_ux_2d + psd_uy_2d
    _, rad_total = radial_spectrum(psd_total_2d)
    viz.plot_radial_spectrum(k_bins, rad_total,
                            title="Radial PSD — total energy",
                            save_name="spectral_radial_total")

    # ---- 峰值检测 ----
    peaks_spatial = detect_peak_wavenumbers(k_bins, rad_total)

    # ---- 时间 PSD ----
    freqs_ux, tpsd_ux = temporal_psd_avg(data, component=0, dt=dt)
    freqs_uy, tpsd_uy = temporal_psd_avg(data, component=1, dt=dt)
    viz.plot_temporal_psd(freqs_ux, tpsd_ux,
                         title="Temporal PSD (spatially averaged) — $u_x$",
                         save_name="spectral_temporal_ux")
    viz.plot_temporal_psd(freqs_uy, tpsd_uy,
                         title="Temporal PSD (spatially averaged) — $u_y$",
                         save_name="spectral_temporal_uy")
    peaks_temporal = detect_peak_frequencies(freqs_ux, tpsd_ux)

    results.update(
        psd_ux_2d=psd_ux_2d, psd_uy_2d=psd_uy_2d,
        psd_total_2d=psd_total_2d,
        k_bins=k_bins,
        rad_ux=rad_ux, rad_uy=rad_uy, rad_total=rad_total,
        peaks_spatial=peaks_spatial,
        freqs_ux=freqs_ux, tpsd_ux=tpsd_ux,
        freqs_uy=freqs_uy, tpsd_uy=tpsd_uy,
        peaks_temporal=peaks_temporal,
    )
    return results
