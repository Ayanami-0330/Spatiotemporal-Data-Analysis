# ME5311 Project 1 — 时空数据分析

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README_EN.md)
[![cn](https://img.shields.io/badge/lang-中文-red.svg)](README.md)

**作者**：虞惠泽

本项目为 **NUS ME5311** 课程 **Project 1** 的数据分析代码仓库。

---

## 项目简介

针对大规模时空矢量场数据集进行**纯数据驱动分析**，综合运用 **SVD/PCA**、**傅里叶谱分析**与**对称性/各向异性诊断**等方法，提取数据中的空间结构、能量分布与时空特征。本项目不涉及系统动力学建模或预测（该部分为 Project 2 内容）。

## 数据集

| 属性 | 值 |
|------|-----|
| 形状 | `(15000, 64, 64, 2)` |
| 描述 | 二维周期域上的双分量矢量场 |
| 时间步长 | `Δt = 0.2` |
| 总时长 | `3000` 仿真时间单位 |
| 文件 | `data/vector_64.npy` |

## 目录结构

```
Spatiotemporal-Data-Analysis/
├── data/                    # 数据集（已忽略）
├── docs/                    # 项目文档（已忽略）
├── figures/                 # 输出图片（已忽略）
├── notebooks/               # 可选 Jupyter notebook
├── src/                     # 分析模块
│   ├── data_loader.py       #   数据加载与预处理
│   ├── svd_analysis.py      #   SVD/PCA 模态分析
│   ├── spectral_analysis.py #   傅里叶/功率谱分析
│   ├── symmetry_analysis.py #   对称性/各向异性诊断
│   └── visualization.py     #   统一可视化工具
├── main.py                  # 一键运行完整分析
├── pyproject.toml           # 项目依赖
├── .gitignore
└── README.md
```

## 主要功能

- **均值场/波动场分离**：去除时间均值，突出动态结构
- **SVD 分析**：奇异值能量谱、主导空间模态、时间系数频谱
- **空间谱分析**：2D FFT 功率谱、径向谱、峰值波数检测
- **时间谱分析**：时间方向 PSD（空间平均）、峰值频率检测
- **对称性诊断**：镜像/旋转对称性检验、各向异性量化
- **衍生物理量**：涡度 ω、散度 ∇·u 计算与分析

## Jupyter Notebooks

`notebooks/` 目录中提供了两个交互式 Jupyter notebook，可逐步探索完整分析流程：

### 英文版本
- **文件**: `notebooks/main_analysis_en.ipynb`
- **内容**: 完整的数据驱动分析工作流及详细解读
- **包含步骤**：
  - Step 0：数据加载与预处理
  - Step 1：SVD 分析（主导空间结构）
  - Step 2：傅里叶谱分析（能量分布与周期性）
  - Step 3：对称性与各向异性诊断
  - 总结：Q1–Q4 项目问题的完整分析结果

### 中文版本
- **文件**: `notebooks/main_analysis.ipynb`
- **内容**: 与英文版本相同的分析流程，中文解释

**使用方式**：使用 Jupyter Lab/Notebook 打开任一 notebook 并按顺序运行单元格：
```bash
jupyter lab notebooks/main_analysis.ipynb
```

所有生成的图像会自动保存至 `figures/` 目录并在单元格中实时显示。

## 依赖

- Python ≥ 3.10
- `numpy` ≥ 1.24
- `matplotlib` ≥ 3.7
- `jupyter` (可选，用于交互式 notebook 分析)

## 安装与运行

### 基础安装（命令行）

```bash
# 安装最小依赖
pip install .

# 运行完整分析
python main.py
```

### 完整安装（含 Jupyter Notebook 支持）

如需使用交互式 Jupyter notebook：

```bash
# 安装可选的 notebook 依赖
pip install .[notebook]

# 启动 notebook
jupyter lab notebooks/main_analysis.ipynb
```

所有图像将自动保存至 `figures/` 目录。
