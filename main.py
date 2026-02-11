"""
Author: Yu Huize (hayes_yu@163.com)
Date: 2026-02-11
Course: NUS ME5311 Project 1
"""

"""
main.py — ME5311 Project 1 主分析脚本
=====================================
串联所有分析模块，按步骤执行完整分析流水线。
所有图片自动保存至 figures/ 目录。

用法:
    cd <project_root>
    python main.py
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


def main():
    print("=" * 60)
    print("  ME5311 Project 1 — Spatio-temporal Data Analysis")
    print("=" * 60)

    # ── Step 0: 数据加载与预处理 ─────────────────────────────
    print("\n▶ Step 0: Loading & preprocessing …")
    bundle = dl.load_and_preprocess()
    raw         = bundle["raw"]           # (15000, 64, 64, 2)
    mean_field  = bundle["mean_field"]    # (64, 64, 2)
    fluctuation = bundle["fluctuation"]   # (15000, 64, 64, 2)
    data_matrix = bundle["data_matrix"]   # (8192, 15000)
    vorticity   = bundle["vorticity"]     # (15000, 64, 64)
    divergence  = bundle["divergence"]    # (15000, 64, 64)

    # 可视化：均值场、示例快照、涡度/散度快照
    viz.plot_vector_snapshot(mean_field, title="Time-averaged mean field",
                            save_name="step0_mean_field")
    viz.plot_vector_snapshot(raw[0], title="Snapshot t=0",
                            save_name="step0_snapshot_t0")
    viz.plot_scalar_field(vorticity[0], title="Vorticity $\\omega$ at t=0",
                          save_name="step0_vorticity_t0")
    viz.plot_scalar_field(divergence[0], title="Divergence $\\nabla\\cdot u$ at t=0",
                          save_name="step0_divergence_t0")

    # ── Step 1: SVD / PCA 分析 ───────────────────────────────
    print("\n▶ Step 1: SVD analysis on fluctuation field …")
    svd_results = svd.run(data_matrix, dt=dl.DT, ny=dl.NY, nx=dl.NX,
                          n_modes=6)

    # ── Step 2: 空间 + 时间谱分析 ────────────────────────────
    print("\n▶ Step 2: Spectral analysis …")
    # 对波动场做谱分析
    spec_results = spectral.run(fluctuation, dt=dl.DT)

    # 补充：对涡度场做空间谱分析
    print("\n  [extra] Vorticity spatial spectrum …")
    psd_vor_2d = np.mean(np.abs(np.fft.fft2(vorticity, axes=(1, 2))) ** 2,
                         axis=0) / (dl.NX * dl.NY)
    k_bins, rad_vor = spectral.radial_spectrum(psd_vor_2d)
    viz.plot_radial_spectrum(k_bins, rad_vor,
                            title="Radial PSD — vorticity",
                            save_name="spectral_radial_vorticity")
    spectral.detect_peak_wavenumbers(k_bins, rad_vor)

    # ── Step 3: 对称性 & 各向异性诊断 ────────────────────────
    print("\n▶ Step 3: Symmetry & anisotropy diagnostics …")
    spatial_modes = svd.extract_spatial_modes(
        svd_results["U"], n_modes=6)
    sym_results = symmetry.run(
        fluctuation, spec_results["psd_total_2d"], svd_modes=spatial_modes)

    # 补充：ux vs uy 分量径向谱对比
    comp = spectral.compare_components_spatial(fluctuation)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(comp["k_bins"], comp["psd_ux_radial"], "o-", ms=3,
                label="$u_x$")
    ax.semilogy(comp["k_bins"], comp["psd_uy_radial"], "s-", ms=3,
                label="$u_y$")
    ax.set_xlabel("Radial wavenumber $k$")
    ax.set_ylabel("PSD($k$)")
    ax.set_title("Component comparison: $u_x$ vs $u_y$ radial spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    viz.savefig(fig, "spectral_component_comparison")

    # ── 汇总 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Analysis complete. Figures saved to:", viz.FIG_DIR)
    print("=" * 60)

    # 返回所有结果（可在交互环境中进一步探索）
    return dict(bundle=bundle, svd=svd_results,
                spectral=spec_results, symmetry=sym_results)


if __name__ == "__main__":
    results = main()
