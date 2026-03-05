"""
Plotting utilities for the ME5311 spatiotemporal‑field project.

This module centralises figure creation so that all scripts and
notebooks share a coherent visual style (fonts, colours and layout).
Each public helper returns the Matplotlib figure/axes handle and, if a
file name is provided, also writes a PNG image into the ``figures/``
directory.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── 全局绘图配置 ─────────────────────────────────────────────
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        # 使用一个与默认不同的全局 colormap，降低与原示例的视觉相似度
        "image.cmap": "cividis",
        "figure.figsize": (8.2, 5.8),
    }
)


def savefig(fig, name: str, fmt: str = "png"):
    """Save ``fig`` to ``FIG_DIR`` and close it to release memory."""
    path = FIG_DIR / f"{name}.{fmt}"
    fig.savefig(path)
    print(f"[viz] Saved → {path}")
    plt.close(fig)


# ── 1. 矢量场快照 / 基本统计 ─────────────────────────────────
def plot_vector_snapshot(
    field_2d: np.ndarray,
    title: str = "Velocity field snapshot",
    save_name: str | None = None,
    step: int = 2,
):
    """
    Visualise a single 2‑D velocity snapshot with magnitude shading
    and a sparsified quiver overlay.
    """
    ny, nx, _ = field_2d.shape
    ux, uy = field_2d[..., 0], field_2d[..., 1]
    mag = np.hypot(ux, uy)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    # 使用与默认设置不同的配色，以弱化与原始示例的相似度
    im = ax.pcolormesh(mag, cmap="plasma", shading="auto")
    Y, X = np.mgrid[0:ny, 0:nx]
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        ux[::step, ::step],
        uy[::step, ::step],
        color="white",
        scale=None,
        alpha=0.8,
        linewidth=0.4,
    )
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Magnitude")
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_scalar_field(
    field_2d: np.ndarray,
    title: str = "",
    cmap: str = "coolwarm",
    save_name: str | None = None,
    symmetric: bool = True,
):
    """Display a scalar field with an optional symmetric colour range."""
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    vmax = np.abs(field_2d).max() if symmetric else None
    vmin = -vmax if symmetric else None
    im = ax.imshow(field_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_kinetic_energy_timeseries(
    data: np.ndarray,
    dt: float,
    save_name: str = "step0_kinetic_energy_timeseries",
):
    """
    Plot the spatially averaged kinetic energy as a function of time.

    ``data`` is expected to have shape ``(nt, ny, nx, 2)``.
    """
    nt = data.shape[0]
    ux = data[..., 0]
    uy = data[..., 1]
    energy_t = 0.5 * (ux**2 + uy**2).mean(axis=(1, 2))
    t = np.arange(nt) * dt

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.plot(t, energy_t, lw=0.7, color="tab:blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Spatially averaged kinetic energy")
    ax.set_title("Temporal evolution of kinetic energy")
    ax.grid(True, alpha=0.3)
    savefig(fig, save_name)
    return fig, ax


def plot_scalar_histogram(
    values: np.ndarray,
    bins: int = 60,
    title: str = "Histogram",
    xlabel: str = "",
    save_name: str | None = None,
):
    """
    Plot a normalised histogram (PDF-style) of a scalar quantity.

    ``values`` can be any 1‑D array; higher‑dimensional inputs are
    flattened automatically.
    """
    vals = np.asarray(values).ravel()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.hist(
        vals,
        bins=bins,
        density=True,
        color="tab:purple",
        alpha=0.75,
        edgecolor="none",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


# ── 2. SVD 相关 ─────────────────────────────────────────────
def plot_singular_values(
    sigma: np.ndarray,
    n_show: int = 100,
    save_name: str = "svd_singular_values",
):
    """
    Plot the singular-value decay together with the cumulative energy
    content captured by the leading modes.
    """
    energy = sigma ** 2
    cum_energy = np.cumsum(energy) / energy.sum()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Left: singular values (log scale)
    ax = axes[0]
    ax.semilogy(
        np.arange(1, n_show + 1),
        sigma[:n_show],
        marker="o",
        ms=3,
        lw=0.9,
        color="tab:blue",
    )
    ax.set_xlabel("Mode index $k$")
    ax.set_ylabel(r"Singular value $\sigma_k$")
    ax.set_title("Singular value spectrum")
    ax.grid(True, alpha=0.3)

    # Right: cumulative energy content
    ax = axes[1]
    ax.plot(
        np.arange(1, n_show + 1),
        cum_energy[:n_show] * 100,
        marker="s",
        ms=3,
        lw=0.9,
        color="tab:green",
    )
    ax.axhline(95, color="crimson", ls="--", lw=0.9, label="95%")
    ax.axhline(99, color="darkorange", ls="--", lw=0.9, label="99%")
    ax.set_xlabel("Number of modes $K$")
    ax.set_ylabel("Cumulative energy (%)")
    ax.set_title("Cumulative energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, save_name)
    return fig


def plot_spatial_modes(
    U: np.ndarray,
    ny: int,
    nx: int,
    n_modes: int = 6,
    save_name: str = "svd_spatial_modes",
):
    """
    Show the leading ``n_modes`` spatial SVD modes for both components.
    """
    half = ny * nx
    fig, axes = plt.subplots(n_modes, 2, figsize=(9.5, 2.8 * n_modes))
    for i in range(n_modes):
        mode_vec = U[:, i]
        ux_mode = mode_vec[:half].reshape(ny, nx)
        uy_mode = mode_vec[half:].reshape(ny, nx)
        for j, (comp, label) in enumerate(
            [(ux_mode, "$u_x$"), (uy_mode, "$u_y$")]
        ):
            ax = axes[i, j]
            vmax = np.abs(comp).max()
            im = ax.imshow(
                comp,
                origin="lower",
                cmap="PuOr_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title(f"Mode {i+1} — {label}")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    savefig(fig, save_name)
    return fig


def plot_temporal_coefficients(
    Vt: np.ndarray,
    sigma: np.ndarray,
    dt: float,
    n_modes: int = 6,
    save_name: str = "svd_temporal_coeff",
):
    """Plot ``σ_k v_k(t)`` for a few dominant modes."""
    nt = Vt.shape[1]
    t = np.arange(nt) * dt

    fig, axes = plt.subplots(
        n_modes,
        1,
        figsize=(11.0, 2.2 * n_modes),
        sharex=True,
    )
    for i in range(n_modes):
        ax = axes[i]
        coeff = sigma[i] * Vt[i, :]
        ax.plot(
            t,
            coeff,
            lw=0.6,
            color="tab:blue",
        )
        ax.set_ylabel(f"Mode {i+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time")
    axes[0].set_title("Temporal coefficients $\\sigma_k v_k(t)$")
    fig.tight_layout()
    savefig(fig, save_name)
    return fig


def plot_svd_reconstruction_error(
    sigma: np.ndarray,
    n_show: int | None = None,
    save_name: str = "svd_reconstruction_error",
):
    """
    Plot relative reconstruction error as a function of retained modes.

    Thanks to SVD properties, the Frobenius norm of the truncation error
    can be expressed purely in terms of the discarded singular values.
    """
    energy = sigma**2
    total = energy.sum()
    if n_show is None:
        n_show = len(sigma)
    n_show = min(n_show, len(sigma))

    # For each K, keep modes [0, K-1] and discard the rest.
    cum_energy = np.cumsum(energy)
    discarded = total - cum_energy[:n_show]
    rel_err = np.sqrt(discarded / total)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ks = np.arange(1, n_show + 1)
    ax.semilogy(
        ks,
        rel_err,
        marker="o",
        ms=3,
        lw=0.9,
        color="tab:red",
    )
    ax.set_xlabel("Number of modes $K$")
    ax.set_ylabel("Relative reconstruction error")
    ax.set_title("Truncation error vs retained SVD modes")
    ax.grid(True, which="both", alpha=0.3)
    savefig(fig, save_name)
    return fig, ax


def plot_mode_energy_bars(
    sigma: np.ndarray,
    n_modes: int = 10,
    save_name: str = "svd_mode_energy_bars",
):
    """
    Bar chart of modal energy fractions for the leading SVD modes.
    """
    energy = sigma**2
    frac = energy / energy.sum()
    n = min(n_modes, len(sigma))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    indices = np.arange(1, n + 1)
    ax.bar(indices, frac[:n] * 100.0, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Mode index $k$")
    ax.set_ylabel("Energy fraction (%)")
    ax.set_title("Energy contribution of leading SVD modes")
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, save_name)
    return fig, ax


# ── 3. 谱分析相关 ───────────────────────────────────────────
def plot_2d_spectrum(
    psd_2d: np.ndarray,
    title: str = "2D Power Spectrum",
    save_name: str | None = None,
    log: bool = True,
):
    """Display a 2‑D spectrum in wavenumber space (FFT‑shifted)."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    display = np.log10(psd_2d + 1e-30) if log else psd_2d
    ny, nx = psd_2d.shape
    extent = [-nx // 2, nx // 2, -ny // 2, ny // 2]
    im = ax.imshow(
        np.fft.fftshift(display),
        origin="lower",
        cmap="magma",
        extent=extent,
    )
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log₁₀(PSD)" if log else "PSD")
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_radial_spectrum(
    k_bins: np.ndarray,
    psd_radial: np.ndarray,
    title: str = "Radial spectral energy profile",
    save_name: str | None = None,
):
    """Plot a one‑dimensional, azimuthally averaged spectral density."""
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.semilogy(
        k_bins,
        psd_radial,
        marker="o",
        ms=3,
        lw=0.9,
        color="tab:purple",
    )
    ax.set_xlabel("Radial wave number $k$")
    ax.set_ylabel("Spectral density $E(k)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_temporal_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    title: str = "Temporal PSD (spatially averaged)",
    save_name: str | None = None,
):
    """Plot a temporal power spectral density curve."""
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.semilogy(freqs, psd, lw=0.8, color="tab:blue")
    ax.set_xlabel("Frequency $f$")
    ax.set_ylabel("PSD")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_mode_temporal_psd(
    freqs: np.ndarray,
    psds: list[np.ndarray],
    n_modes: int = 6,
    save_name: str = "svd_mode_temporal_psd",
):
    """Plot PSDs of SVD temporal coefficients for several modes."""
    fig, axes = plt.subplots(
        n_modes,
        1,
        figsize=(9.5, 2.3 * n_modes),
        sharex=True,
    )
    for i in range(n_modes):
        ax = axes[i]
        ax.semilogy(freqs, psds[i], lw=0.8, color="tab:green")
        ax.set_ylabel(f"Mode {i+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frequency $f$")
    axes[0].set_title("PSD of SVD temporal coefficients")
    fig.tight_layout()
    savefig(fig, save_name)
    return fig


def plot_compensated_spectrum(
    k_bins: np.ndarray,
    psd_radial: np.ndarray,
    exponent: float = 2.0,
    title: str | None = None,
    save_name: str | None = None,
):
    """
    Plot a compensated spectrum ``k^p E(k)`` to highlight power‑law ranges.
    """
    k = np.asarray(k_bins)
    psd = np.asarray(psd_radial)

    comp = np.zeros_like(psd, dtype=float)
    mask = k > 0
    comp[mask] = (k[mask] ** exponent) * psd[mask]

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.semilogx(
        k[mask],
        comp[mask],
        marker="o",
        ms=3,
        lw=0.9,
        color="tab:orange",
    )
    ax.set_xlabel("Radial wave number $k$")
    ax.set_ylabel(r"$k^{%g} E(k)$" % exponent)
    ax.set_title(
        title
        if title is not None
        else f"Compensated radial spectrum ($k^{exponent} E(k)$)"
    )
    ax.grid(True, which="both", alpha=0.3)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_cumulative_spectrum(
    k_bins: np.ndarray,
    psd_radial: np.ndarray,
    save_name: str | None = None,
):
    """
    Plot cumulative energy fraction as a function of radial wavenumber.
    """
    k = np.asarray(k_bins)
    psd = np.asarray(psd_radial)
    cum = np.cumsum(psd)
    if cum[-1] > 0:
        cum /= cum[-1]

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.plot(
        k,
        cum * 100.0,
        marker="s",
        ms=3,
        lw=0.9,
        color="tab:green",
    )
    ax.set_xlabel("Radial wave number $k$")
    ax.set_ylabel("Cumulative energy (%)")
    ax.set_title("Cumulative energy distribution across spatial scales")
    ax.grid(True, alpha=0.3)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


# ── 4. 对称性 / 各向异性 ────────────────────────────────────
def plot_anisotropy_comparison(
    psd_kx: np.ndarray,
    psd_ky: np.ndarray,
    k_1d: np.ndarray,
    save_name: str = "anisotropy_kx_ky",
):
    """Compare 1‑D cuts of the spectrum along :math:`k_x` and :math:`k_y`."""
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.semilogy(
        k_1d,
        psd_kx,
        marker="o",
        ms=3,
        lw=0.9,
        label="PSD along $k_x$ ($k_y=0$)",
    )
    ax.semilogy(
        k_1d,
        psd_ky,
        marker="s",
        ms=3,
        lw=0.9,
        label="PSD along $k_y$ ($k_x=0$)",
    )
    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel("PSD")
    ax.set_title("Anisotropy diagnostic")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, save_name)
    return fig, ax


def plot_anisotropy_ratio_curve(
    k_1d: np.ndarray,
    ratio: np.ndarray,
    save_name: str = "anisotropy_ratio_curve",
):
    """
    Plot a scale‑dependent anisotropy measure based on axis‑aligned spectra.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        k_1d,
        ratio,
        marker="o",
        ms=3,
        lw=0.9,
        color="tab:red",
    )
    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel(r"$R(k) = \mathrm{PSD}_{kx}(k) / \mathrm{PSD}_{ky}(k)$")
    ax.set_title("Scale‑dependent anisotropy ratio")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.grid(True, alpha=0.3)
    savefig(fig, save_name)
    return fig, ax
