"""
Author: Yu Huize (hayes_yu@163.com)
Date: 2026-02-11
Course: NUS ME5311 Project 1
"""

"""
visualization.py — 统一可视化工具
=================================
为各分析模块提供绘图函数，统一配色、标注、导出。
所有 figure 默认保存至 figures/ 目录。
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── 全局配置 ─────────────────────────────────────────────────
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "image.cmap":       "RdBu_r",
    "figure.figsize":   (8, 6),
})


def savefig(fig, name: str, fmt: str = "png"):
    """保存图片到 figures/ 目录"""
    path = FIG_DIR / f"{name}.{fmt}"
    fig.savefig(path)
    print(f"[viz] Saved → {path}")
    plt.close(fig)


# ── 1. 矢量场快照 ───────────────────────────────────────────
def plot_vector_snapshot(field_2d: np.ndarray, title: str = "Vector field",
                         save_name: str | None = None, step: int = 2):
    """
    绘制单个 2D 矢量场快照。
    field_2d: (ny, nx, 2)
    step: 箭头间隔（降采样，避免过密）
    """
    ny, nx, _ = field_2d.shape
    ux, uy = field_2d[..., 0], field_2d[..., 1]
    mag = np.sqrt(ux**2 + uy**2)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.pcolormesh(mag, cmap="viridis", shading="auto")
    Y, X = np.mgrid[0:ny, 0:nx]
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              ux[::step, ::step], uy[::step, ::step],
              color="k", scale=None, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Magnitude")
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_scalar_field(field_2d: np.ndarray, title: str = "",
                      cmap: str = "RdBu_r", save_name: str | None = None,
                      symmetric: bool = True):
    """绘制标量场 (ny, nx)，可选对称色标。"""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    vmax = np.abs(field_2d).max() if symmetric else None
    vmin = -vmax if symmetric else None
    im = ax.imshow(field_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


# ── 2. SVD 相关 ─────────────────────────────────────────────
def plot_singular_values(sigma: np.ndarray, n_show: int = 100,
                         save_name: str = "svd_singular_values"):
    """奇异值衰减曲线 + 累积能量占比。"""
    energy = sigma**2
    cum_energy = np.cumsum(energy) / energy.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：奇异值（对数坐标）
    ax = axes[0]
    ax.semilogy(np.arange(1, n_show + 1), sigma[:n_show], "o-", ms=3)
    ax.set_xlabel("Mode index $k$")
    ax.set_ylabel(r"Singular value $\sigma_k$")
    ax.set_title("Singular value spectrum")
    ax.grid(True, alpha=0.3)

    # 右图：累积能量
    ax = axes[1]
    ax.plot(np.arange(1, n_show + 1), cum_energy[:n_show] * 100, "s-", ms=3)
    ax.axhline(95, color="r", ls="--", lw=1, label="95%")
    ax.axhline(99, color="orange", ls="--", lw=1, label="99%")
    ax.set_xlabel("Number of modes $K$")
    ax.set_ylabel("Cumulative energy (%)")
    ax.set_title("Cumulative energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, save_name)
    return fig


def plot_spatial_modes(U: np.ndarray, ny: int, nx: int,
                       n_modes: int = 6,
                       save_name: str = "svd_spatial_modes"):
    """
    可视化前 n_modes 个 SVD 空间模态。
    U: (N, K)  N = 2*ny*nx，前 N//2 为 ux，后 N//2 为 uy。
    """
    half = ny * nx
    fig, axes = plt.subplots(n_modes, 2, figsize=(10, 3 * n_modes))
    for i in range(n_modes):
        mode = U[:, i]
        ux_mode = mode[:half].reshape(ny, nx)
        uy_mode = mode[half:].reshape(ny, nx)
        for j, (comp, label) in enumerate([(ux_mode, "$u_x$"), (uy_mode, "$u_y$")]):
            ax = axes[i, j]
            vmax = np.abs(comp).max()
            im = ax.imshow(comp, origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            ax.set_title(f"Mode {i+1} — {label}")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    savefig(fig, save_name)
    return fig


def plot_temporal_coefficients(Vt: np.ndarray, sigma: np.ndarray,
                                dt: float, n_modes: int = 6,
                                save_name: str = "svd_temporal_coeff"):
    """绘制前 n_modes 个模态的时间系数 σ_k * v_k(t)。"""
    nt = Vt.shape[1]
    t = np.arange(nt) * dt

    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 2.5 * n_modes),
                             sharex=True)
    for i in range(n_modes):
        ax = axes[i]
        coeff = sigma[i] * Vt[i, :]
        ax.plot(t, coeff, lw=0.5)
        ax.set_ylabel(f"Mode {i+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time")
    axes[0].set_title("Temporal coefficients $\\sigma_k v_k(t)$")
    fig.tight_layout()
    savefig(fig, save_name)
    return fig


# ── 3. 谱分析相关 ───────────────────────────────────────────
def plot_2d_spectrum(psd_2d: np.ndarray, title: str = "2D Power Spectrum",
                     save_name: str | None = None, log: bool = True):
    """绘制 2D 功率谱（波数域），中心化显示。"""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    display = np.log10(psd_2d + 1e-30) if log else psd_2d
    ny, nx = psd_2d.shape
    extent = [-nx // 2, nx // 2, -ny // 2, ny // 2]
    im = ax.imshow(np.fft.fftshift(display), origin="lower",
                   cmap="inferno", extent=extent)
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log₁₀(PSD)" if log else "PSD")
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_radial_spectrum(k_bins: np.ndarray, psd_radial: np.ndarray,
                         title: str = "Radial Power Spectrum",
                         save_name: str | None = None):
    """绘制径向（1D）功率谱。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(k_bins, psd_radial, "o-", ms=3)
    ax.set_xlabel("Radial wavenumber $k$")
    ax.set_ylabel("PSD($k$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_temporal_psd(freqs: np.ndarray, psd: np.ndarray,
                      title: str = "Temporal PSD (spatially averaged)",
                      save_name: str | None = None):
    """时间频率功率谱。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(freqs, psd, lw=0.8)
    ax.set_xlabel("Frequency $f$")
    ax.set_ylabel("PSD")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_mode_temporal_psd(freqs: np.ndarray, psds: list[np.ndarray],
                           n_modes: int = 6,
                           save_name: str = "svd_mode_temporal_psd"):
    """前 K 个 SVD 模态时间系数的 PSD。"""
    fig, axes = plt.subplots(n_modes, 1, figsize=(10, 2.5 * n_modes),
                             sharex=True)
    for i in range(n_modes):
        ax = axes[i]
        ax.semilogy(freqs, psds[i], lw=0.8)
        ax.set_ylabel(f"Mode {i+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frequency $f$")
    axes[0].set_title("PSD of SVD temporal coefficients")
    fig.tight_layout()
    savefig(fig, save_name)
    return fig


# ── 4. 对称性 / 各向异性 ────────────────────────────────────
def plot_anisotropy_comparison(psd_kx: np.ndarray, psd_ky: np.ndarray,
                                k_1d: np.ndarray,
                                save_name: str = "anisotropy_kx_ky"):
    """对比 kx 方向与 ky 方向的 1D 谱切片。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(k_1d, psd_kx, "o-", ms=3, label="PSD along $k_x$ ($k_y=0$)")
    ax.semilogy(k_1d, psd_ky, "s-", ms=3, label="PSD along $k_y$ ($k_x=0$)")
    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel("PSD")
    ax.set_title("Anisotropy diagnostic")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, save_name)
    return fig, ax
