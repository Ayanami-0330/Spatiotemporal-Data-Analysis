"""
Author: Yu Huize (hayes_yu@163.com)
Date: 2026-02-11
Course: NUS ME5311 Project 1
"""

"""
symmetry_analysis.py — 对称性 & 各向异性诊断
=============================================
- 2D 傅里叶谱沿 kx / ky 轴切片对比
- 谱的镜像对称性检验
- SVD 模态空间对称性检验
- ux / uy 分量径向谱差异量化
"""

import numpy as np
from . import data_loader as dl
from . import visualization as viz


# ── 1. 傅里叶谱各向异性 ─────────────────────────────────────
def axis_slices(psd_2d: np.ndarray):
    """
    沿 kx 轴（ky=0）和 ky 轴（kx=0）切片。
    返回 (k_1d, psd_kx_slice, psd_ky_slice)
    psd_2d 应为 fftshift 之前的原始排布 (ny, nx)。
    """
    ny, nx = psd_2d.shape
    psd_kx = psd_2d[0, :]          # ky=0 行 → 沿 kx
    psd_ky = psd_2d[:, 0]          # kx=0 列 → 沿 ky
    k_1d = np.arange(nx // 2 + 1)  # 非负波数
    # 由于 FFT 输出前半为 [0..N/2]，后半为负频率对称
    psd_kx = psd_kx[:nx // 2 + 1]
    psd_ky = psd_ky[:ny // 2 + 1]
    return k_1d, psd_kx, psd_ky


def anisotropy_ratio(psd_kx: np.ndarray, psd_ky: np.ndarray):
    """
    计算各向异性比  R(k) = PSD_kx(k) / PSD_ky(k)。
    R ≈ 1 → 各向同性；偏离 1 → 各向异性。
    """
    eps = 1e-30
    ratio = psd_kx / (psd_ky + eps)
    return ratio


# ── 2. 镜像对称性检验 ───────────────────────────────────────
def mirror_symmetry(psd_2d: np.ndarray):
    """
    检验 2D PSD 的镜像对称性：
      x-对称: PSD(kx, ky) vs PSD(-kx, ky)
      y-对称: PSD(kx, ky) vs PSD(kx, -ky)
    返回相对误差 (err_x, err_y)，越小越对称。
    """
    # PSD(-kx, ky) = np.flip(PSD, axis=1)（FFT 对称性）
    flip_x = np.flip(psd_2d, axis=1)
    flip_y = np.flip(psd_2d, axis=0)

    norm = np.sum(psd_2d ** 2)
    err_x = np.sum((psd_2d - flip_x) ** 2) / norm
    err_y = np.sum((psd_2d - flip_y) ** 2) / norm
    print(f"[symmetry] Mirror-x relative error: {err_x:.6e}")
    print(f"[symmetry] Mirror-y relative error: {err_y:.6e}")
    return err_x, err_y


def rotational_symmetry_90(psd_2d: np.ndarray):
    """
    检验 90° 旋转对称性：PSD(kx,ky) vs PSD(ky,kx)。
    返回相对误差，越小说明旋转对称性越好（各向同性的必要条件）。
    """
    rotated = psd_2d.T
    norm = np.sum(psd_2d ** 2)
    err = np.sum((psd_2d - rotated) ** 2) / norm
    print(f"[symmetry] 90°-rotation relative error: {err:.6e}")
    return err


# ── 3. SVD 模态对称性 ───────────────────────────────────────
def mode_symmetry_check(mode_2d: np.ndarray):
    """
    对单个空间模态 (ny, nx) 检查：
      - x 方向镜像对称  mode(y, x) vs mode(y, nx-1-x)
      - y 方向镜像对称  mode(y, x) vs mode(ny-1-y, x)
    返回 (corr_x, corr_y)  相关系数，+1=对称, -1=反对称, 0=无关。
    """
    flip_x = np.flip(mode_2d, axis=1)
    flip_y = np.flip(mode_2d, axis=0)

    def _corr(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        return np.corrcoef(a_flat, b_flat)[0, 1]

    corr_x = _corr(mode_2d, flip_x)
    corr_y = _corr(mode_2d, flip_y)
    return corr_x, corr_y


# ── 4. 分量差异量化 ─────────────────────────────────────────
def component_energy_ratio(data: np.ndarray):
    """
    计算 ux 与 uy 的全局能量比。
    返回 (E_ux, E_uy, ratio = E_ux/E_uy)
    """
    E_ux = np.mean(data[..., 0] ** 2)
    E_uy = np.mean(data[..., 1] ** 2)
    ratio = E_ux / E_uy
    print(f"[symmetry] Energy: E_ux={E_ux:.4e}, E_uy={E_uy:.4e}, ratio={ratio:.4f}")
    return E_ux, E_uy, ratio


# ── 顶层运行函数 ────────────────────────────────────────────
def run(data: np.ndarray, psd_total_2d: np.ndarray,
        svd_modes: list | None = None):
    """
    完整对称性/各向异性诊断流水线。
    data         : (nt, ny, nx, 2)  原始或波动场
    psd_total_2d : (ny, nx)  总能量 2D PSD（来自 spectral_analysis）
    svd_modes    : [(ux_2d, uy_2d), ...] 来自 svd_analysis
    """
    results = {}

    # 分量能量比
    E_ux, E_uy, ratio = component_energy_ratio(data)
    results["energy_ratio"] = ratio

    # 傅里叶谱对称性
    err_x, err_y = mirror_symmetry(psd_total_2d)
    err_rot = rotational_symmetry_90(psd_total_2d)
    results["mirror_err_x"] = err_x
    results["mirror_err_y"] = err_y
    results["rotation_err"] = err_rot

    # kx / ky 轴切片各向异性
    k_1d, psd_kx, psd_ky = axis_slices(psd_total_2d)
    viz.plot_anisotropy_comparison(psd_kx, psd_ky, k_1d)
    results["anisotropy_ratio"] = anisotropy_ratio(psd_kx, psd_ky)

    # SVD 模态对称性
    if svd_modes is not None:
        sym_results = []
        for i, (ux_m, uy_m) in enumerate(svd_modes):
            cx_ux, cy_ux = mode_symmetry_check(ux_m)
            cx_uy, cy_uy = mode_symmetry_check(uy_m)
            sym_results.append(dict(
                mode=i + 1,
                ux_corr_x=cx_ux, ux_corr_y=cy_ux,
                uy_corr_x=cx_uy, uy_corr_y=cy_uy,
            ))
            print(f"[symmetry] Mode {i+1}: "
                  f"ux(mirror-x={cx_ux:+.3f}, mirror-y={cy_ux:+.3f}), "
                  f"uy(mirror-x={cx_uy:+.3f}, mirror-y={cy_uy:+.3f})")
        results["mode_symmetry"] = sym_results

    return results
