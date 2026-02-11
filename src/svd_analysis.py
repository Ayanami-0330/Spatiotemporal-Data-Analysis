"""
svd_analysis.py — SVD / PCA 模态分析
====================================
- 经济 SVD 分解
- 奇异值谱 & 累积能量
- 空间模态提取与可视化
- 时间系数提取 + 时间频谱分析
"""

import numpy as np
from . import data_loader as dl
from . import visualization as viz


def perform_svd(data_matrix: np.ndarray, full_matrices: bool = False):
    """
    对数据矩阵 (N, T) 执行经济 SVD。
    返回 U (N, K), sigma (K,), Vt (K, T)  其中 K = min(N, T)
    """
    print("[svd] Computing economy SVD …")
    U, sigma, Vt = np.linalg.svd(data_matrix, full_matrices=full_matrices)
    print(f"[svd] Done. U={U.shape}, sigma={sigma.shape}, Vt={Vt.shape}")
    return U, sigma, Vt


def energy_spectrum(sigma: np.ndarray):
    """
    计算模态能量谱。
    返回:
        energy       : σ_k² 
        cum_energy   : 累积能量占比 (0~1)
        n_95, n_99   : 累积能量达到 95% / 99% 所需模态数
    """
    energy = sigma ** 2
    cum = np.cumsum(energy) / energy.sum()
    n_95 = int(np.searchsorted(cum, 0.95)) + 1
    n_99 = int(np.searchsorted(cum, 0.99)) + 1
    print(f"[svd] Modes for 95% energy: {n_95},  99%: {n_99}")
    return energy, cum, n_95, n_99


def extract_spatial_modes(U: np.ndarray, ny: int = dl.NY, nx: int = dl.NX,
                          n_modes: int = 6):
    """
    提取前 n_modes 个空间模态，reshape 为 (ny, nx) 分量。
    返回列表 [(ux_mode, uy_mode), ...]
    """
    half = ny * nx
    modes = []
    for i in range(n_modes):
        ux = U[:half, i].reshape(ny, nx)
        uy = U[half:, i].reshape(ny, nx)
        modes.append((ux, uy))
    return modes


def temporal_coefficients(sigma: np.ndarray, Vt: np.ndarray,
                          n_modes: int = 6):
    """返回前 n_modes 个模态的时间系数 a_k(t) = σ_k * v_k(t)。"""
    return [sigma[i] * Vt[i, :] for i in range(n_modes)]


def temporal_coefficient_psd(sigma: np.ndarray, Vt: np.ndarray,
                             dt: float = dl.DT, n_modes: int = 6):
    """
    对前 n_modes 个 SVD 时间系数做 FFT，返回 (freqs, [psd_1, …, psd_K])。
    """
    nt = Vt.shape[1]
    freqs = np.fft.rfftfreq(nt, d=dt)
    psds = []
    for i in range(n_modes):
        coeff = sigma[i] * Vt[i, :]
        fhat = np.fft.rfft(coeff)
        psd = (np.abs(fhat) ** 2) / nt
        psds.append(psd)
    return freqs, psds


# ── 顶层运行函数 ────────────────────────────────────────────
def run(data_matrix: np.ndarray, dt: float = dl.DT,
        ny: int = dl.NY, nx: int = dl.NX, n_modes: int = 6):
    """
    完整 SVD 分析流水线：分解 → 能量谱 → 空间模态可视化
    → 时间系数可视化 → 时间系数 PSD。
    """
    U, sigma, Vt = perform_svd(data_matrix)

    # 能量谱
    energy, cum, n95, n99 = energy_spectrum(sigma)
    viz.plot_singular_values(sigma, n_show=min(100, len(sigma)))

    # 空间模态
    viz.plot_spatial_modes(U, ny, nx, n_modes=n_modes)

    # 时间系数
    viz.plot_temporal_coefficients(Vt, sigma, dt, n_modes=n_modes)

    # 时间系数 PSD
    freqs, psds = temporal_coefficient_psd(sigma, Vt, dt, n_modes=n_modes)
    viz.plot_mode_temporal_psd(freqs, psds, n_modes=n_modes)

    return dict(U=U, sigma=sigma, Vt=Vt,
                energy=energy, cum_energy=cum,
                n95=n95, n99=n99, freqs=freqs, psds=psds)
