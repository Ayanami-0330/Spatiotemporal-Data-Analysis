"""
data_loader.py — 数据加载与预处理
=================================
- 加载 vector_64.npy  →  shape (nt, ny, nx, 2)
- 计算时间均值场  mean_field  →  (ny, nx, 2)
- 计算波动场       fluctuation →  (nt, ny, nx, 2)
- 构建数据矩阵   data_matrix  →  (N, T)  N=8192, T=15000
- 计算衍生物理量：涡度 ω、散度 ∇·u（周期边界有限差分）
"""

from pathlib import Path
import numpy as np

# ── 数据集参数 ──────────────────────────────────────────────
NX, NY = 64, 64              # 空间网格分辨率
NT     = 15000               # 时间快照数
DT     = 0.2                 # 时间采样间隔（仿真时间单位）
N_DOF  = NX * NY * 2         # 单快照自由度  8192
T_TOTAL = NT * DT            # 总时长 3000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw(fname: str = "vector_64.npy") -> np.ndarray:
    """加载原始数据 → (nt, ny, nx, 2)"""
    data = np.load(DATA_DIR / fname)
    assert data.shape == (NT, NY, NX, 2), f"Unexpected shape {data.shape}"
    print(f"[data_loader] Loaded {fname}  shape={data.shape}  "
          f"dtype={data.dtype}  size={data.nbytes / 1e9:.2f} GB")
    return data


def compute_mean_field(data: np.ndarray) -> np.ndarray:
    """时间平均场 ū(x,y) → (ny, nx, 2)"""
    return data.mean(axis=0)


def compute_fluctuation(data: np.ndarray,
                        mean_field: np.ndarray | None = None) -> np.ndarray:
    """波动场 u' = u - ū → (nt, ny, nx, 2)"""
    if mean_field is None:
        mean_field = compute_mean_field(data)
    return data - mean_field[np.newaxis, ...]


def build_data_matrix(field: np.ndarray) -> np.ndarray:
    """
    将 4-D 场 (nt, ny, nx, 2) 展平为 2-D 数据矩阵 (N, T)。
    每列 = 一个时间快照的 8192 维向量。
    展平顺序：先 ux 全部 ny×nx，再 uy 全部 ny×nx。
    """
    nt = field.shape[0]
    # (nt, ny, nx, 2) → (nt, 2, ny, nx) → (nt, 2*ny*nx)
    mat = field.transpose(0, 3, 1, 2).reshape(nt, -1)  # (T, N)
    return mat.T  # (N, T)


# ── 衍生物理量（周期边界有限差分） ──────────────────────────
def _periodic_diff(arr: np.ndarray, axis: int, dx: float = 1.0) -> np.ndarray:
    """中心差分（周期边界），axis 指定对哪个轴差分。"""
    return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2.0 * dx)


def compute_vorticity(data: np.ndarray, L: float | None = None) -> np.ndarray:
    """
    涡度  ω = ∂u_y/∂x - ∂u_x/∂y  → (nt, ny, nx)
    data: (nt, ny, nx, 2)  分量顺序 [ux, uy]
    L: 域尺寸，用于计算 dx = L/NX（若未给定则取 dx=1）
    """
    dx = (L / NX) if L is not None else 1.0
    dy = (L / NY) if L is not None else 1.0
    ux = data[..., 0]  # (nt, ny, nx)
    uy = data[..., 1]
    # ∂u_y/∂x  — x 方向对应 axis=2
    duy_dx = _periodic_diff(uy, axis=2, dx=dx)
    # ∂u_x/∂y  — y 方向对应 axis=1
    dux_dy = _periodic_diff(ux, axis=1, dx=dy)
    return duy_dx - dux_dy


def compute_divergence(data: np.ndarray, L: float | None = None) -> np.ndarray:
    """
    散度  ∇·u = ∂u_x/∂x + ∂u_y/∂y  → (nt, ny, nx)
    """
    dx = (L / NX) if L is not None else 1.0
    dy = (L / NY) if L is not None else 1.0
    ux = data[..., 0]
    uy = data[..., 1]
    dux_dx = _periodic_diff(ux, axis=2, dx=dx)
    duy_dy = _periodic_diff(uy, axis=1, dx=dy)
    return dux_dx + duy_dy


# ── 快捷入口 ────────────────────────────────────────────────
def load_and_preprocess(fname: str = "vector_64.npy"):
    """
    一步完成加载 + 均值/波动分离 + 矩阵构建。
    返回 dict:
        raw        : (nt, ny, nx, 2)
        mean_field : (ny, nx, 2)
        fluctuation: (nt, ny, nx, 2)
        data_matrix: (N, T)   基于波动场
        vorticity  : (nt, ny, nx)
        divergence : (nt, ny, nx)
    """
    raw = load_raw(fname)
    mf  = compute_mean_field(raw)
    flu = compute_fluctuation(raw, mf)
    mat = build_data_matrix(flu)
    vor = compute_vorticity(raw)
    div = compute_divergence(raw)
    print(f"[data_loader] data_matrix shape = {mat.shape}")
    print(f"[data_loader] vorticity   shape = {vor.shape}")
    print(f"[data_loader] divergence  shape = {div.shape}")
    return dict(
        raw=raw, mean_field=mf, fluctuation=flu,
        data_matrix=mat, vorticity=vor, divergence=div,
    )
