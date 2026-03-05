"""
Symmetry and anisotropy diagnostics for the vector field.

The routines below quantify, in a few complementary ways,

* how symmetric the 2‑D spectrum is under reflections/90° rotations, and
* how differently the two velocity components behave energetically.
"""

import numpy as np
from . import data_loader as dl
from . import visualization as viz


# ── 1. 傅里叶谱各向异性 ─────────────────────────────────────
def axis_slices(psd_2d: np.ndarray):
    """
    Extract one‑dimensional cuts of the spectrum along :math:`k_x` and
    :math:`k_y` (with the other wavenumber set to zero).

    The input is the unshifted 2‑D PSD of shape ``(ny, nx)``.
    """
    ny, nx = psd_2d.shape
    psd_kx = psd_2d[0, :]          # ky = 0 row → along kx
    psd_ky = psd_2d[:, 0]          # kx = 0 column → along ky
    k_1d = np.arange(nx // 2 + 1)  # non‑negative wavenumbers
    psd_kx = psd_kx[:nx // 2 + 1]
    psd_ky = psd_ky[:ny // 2 + 1]
    return k_1d, psd_kx, psd_ky


def anisotropy_ratio(psd_kx: np.ndarray, psd_ky: np.ndarray):
    """
    Form the scale‑dependent ratio
    :math:`R(k) = \\mathrm{PSD}_{kx}(k) / \\mathrm{PSD}_{ky}(k)`.
    """
    eps = 1e-30
    return psd_kx / (psd_ky + eps)


# ── 2. 镜像对称性检验 ───────────────────────────────────────
def mirror_symmetry(psd_2d: np.ndarray):
    """
    Quantify mirror symmetry of a 2‑D spectrum about both axes.

    Returns relative errors for reflection about :math:`k_x` and
    :math:`k_y`. Smaller values indicate better symmetry.
    """
    flip_x = np.flip(psd_2d, axis=1)  # PSD(-kx, ky)
    flip_y = np.flip(psd_2d, axis=0)  # PSD(kx, -ky)

    norm = np.sum(psd_2d ** 2)
    err_x = np.sum((psd_2d - flip_x) ** 2) / norm
    err_y = np.sum((psd_2d - flip_y) ** 2) / norm
    print(f"[symmetry] mirror-x mismatch (rel.): {err_x:.6e}")
    print(f"[symmetry] mirror-y mismatch (rel.): {err_y:.6e}")
    return err_x, err_y


def rotational_symmetry_90(psd_2d: np.ndarray):
    """
    Compare :math:`\\mathrm{PSD}(k_x, k_y)` with
    :math:`\\mathrm{PSD}(k_y, k_x)` to test 90° rotational symmetry.
    """
    rotated = psd_2d.T
    norm = np.sum(psd_2d ** 2)
    err = np.sum((psd_2d - rotated) ** 2) / norm
    print(f"[symmetry] 90deg-rotation mismatch (rel.): {err:.6e}")
    return err


# ── 3. SVD 模态对称性 ───────────────────────────────────────
def mode_symmetry_check(mode_2d: np.ndarray):
    """
    For a single spatial mode ``(ny, nx)``, compute correlation with its
    horizontally and vertically flipped versions.
    """
    flip_x = np.flip(mode_2d, axis=1)
    flip_y = np.flip(mode_2d, axis=0)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.ravel()
        b_flat = b.ravel()
        return float(np.corrcoef(a_flat, b_flat)[0, 1])

    corr_x = _corr(mode_2d, flip_x)
    corr_y = _corr(mode_2d, flip_y)
    return corr_x, corr_y


# ── 4. 分量差异量化 ─────────────────────────────────────────
def component_energy_ratio(data: np.ndarray):
    """
    Compute global kinetic energy of :math:`u_x` and :math:`u_y`.

    Returns energies and their ratio ``E_ux / E_uy``.
    """
    E_ux = np.mean(data[..., 0] ** 2)
    E_uy = np.mean(data[..., 1] ** 2)
    ratio = E_ux / E_uy
    print(
        f"[symmetry] Energy: "
        f"E_ux={E_ux:.4e}, E_uy={E_uy:.4e}, ratio={ratio:.4f}"
    )
    return E_ux, E_uy, ratio


# ── 顶层运行函数 ────────────────────────────────────────────
def symmetry_overview(
    data: np.ndarray,
    psd_total_2d: np.ndarray,
    svd_modes: list | None = None,
):
    """
    Collect a compact set of symmetry/anisotropy indicators.

    Parameters
    ----------
    data : ndarray
        Vector field (original or fluctuation) with shape
        ``(nt, ny, nx, 2)``.
    psd_total_2d : ndarray
        Total 2‑D PSD from the spectral module, shape ``(ny, nx)``.
    svd_modes : list of tuple(ndarray, ndarray), optional
        Spatial SVD modes ``[(ux_mode, uy_mode), ...]``.
    """
    results: dict[str, np.ndarray | list[dict[str, float]] | float] = {}

    # Component‑wise energy content.
    _, _, ratio = component_energy_ratio(data)
    results["energy_ratio"] = ratio

    # Symmetry of the 2‑D spectrum.
    err_x, err_y = mirror_symmetry(psd_total_2d)
    err_rot = rotational_symmetry_90(psd_total_2d)
    results["mirror_err_x"] = err_x
    results["mirror_err_y"] = err_y
    results["rotation_err"] = err_rot

    # Anisotropy along principal axes in k‑space.
    k_1d, psd_kx, psd_ky = axis_slices(psd_total_2d)
    viz.plot_anisotropy_comparison(psd_kx, psd_ky, k_1d)
    ratio = anisotropy_ratio(psd_kx, psd_ky)
    viz.plot_anisotropy_ratio_curve(k_1d, ratio)
    results["anisotropy_ratio"] = ratio

    # Symmetry properties of individual SVD modes, if provided.
    if svd_modes is not None:
        sym_results: list[dict[str, float]] = []
        for i, (ux_m, uy_m) in enumerate(svd_modes):
            cx_ux, cy_ux = mode_symmetry_check(ux_m)
            cx_uy, cy_uy = mode_symmetry_check(uy_m)
            sym_results.append(
                dict(
                    mode=i + 1,
                    ux_corr_x=cx_ux,
                    ux_corr_y=cy_ux,
                    uy_corr_x=cx_uy,
                    uy_corr_y=cy_uy,
                )
            )
            print(
                f"[symmetry] Mode {i+1}: "
                f"ux(mirror-x={cx_ux:+.3f}, mirror-y={cy_ux:+.3f}), "
                f"uy(mirror-x={cx_uy:+.3f}, mirror-y={cy_uy:+.3f})"
            )
        results["mode_symmetry"] = sym_results

    return results


def run(
    data: np.ndarray,
    psd_total_2d: np.ndarray,
    svd_modes: list | None = None,
):
    """
    Backwards‑compatible wrapper around :func:`symmetry_overview`.
    """
    return symmetry_overview(
        data=data,
        psd_total_2d=psd_total_2d,
        svd_modes=svd_modes,
    )
