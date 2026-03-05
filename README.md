# ME5311 Project 1 — 时空数据分析（个人实现）

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README_EN.md)
[![cn](https://img.shields.io/badge/lang-中文-red.svg)](README.md)

**作者**：谭开来（e1554333@u.nus.edu）  
**课程**：NUS ME5311 — Data‑driven modelling and control

本仓库是我针对 **ME5311 Project 1** 所做的**完整数据分析实现**，只使用课程方
提供的时空矢量场数据，不额外构建动力学模型，也不做预测。所有结论都来自对
数据本身的结构与统计特征的挖掘。

---

## 1. 作业背景与目标

Project 1 的官方说明希望我们围绕下面几个问题，对数据做“数据驱动”的探索：

1. **Q1 — 主导空间结构是什么？**  
   能否通过 SVD 等方法识别出少数几个主导模态？
2. **Q2 — 能量在不同空间尺度上的分布如何？**  
   哪些波数（长度尺度）携带了最多能量？
3. **Q3 — 能否从数据中反推出外加周期强迫的波数？**  
   在傅里叶谱中是否能看到清晰的特征波数峰值？
4. **Q4 — 系统是否各向同性 / 具有哪些对称性或各向异性？**

本仓库中的代码结构，就是按这四个问题设计的：每一部分分析都对应到至少一个
问题，并在最终报告中的综合图中有所体现。

---

## 2. 数据集概况

项目只包含一个主数据集：

| 属性     | 值                           |
|----------|------------------------------|
| 形状     | `(15000, 64, 64, 2)`        |
| 描述     | 周期二维区域上的双分量矢量场 |
| 时间步长 | `Δt = 0.2`                  |
| 总时长   | `3000` 仿真时间单位          |
| 文件     | `data/vector_64.npy`        |

每一个快照记录了 `64×64` 网格上的两个速度分量，构成一个高维状态向量。后续所有
分析都直接基于这一数组。

---

## 3. 分析流程概览

整体流程分为四个阶段，与课堂内容基本对应：

1. **Step 0：预处理与基本统计**  
   - 加载矢量场，拆分为时间平均的背景场与波动场；  
   - 计算涡度与散度场，并绘制代表性快照；  
   - 统计并绘制全局动能时间序列；  
   - 对涡度 / 散度做直方图，观察概率分布形状。

2. **Step 1：波动场的低秩 SVD 分解**  
   - 对快照矩阵做经济型 SVD；  
   - 分析奇异值谱与累计能量；  
   - 额外绘制**重构误差随模态数量变化曲线**，作为选取模态数的依据；  
   - 可视化前若干空间模态及其时间系数。

3. **Step 2：空间与时间谱特性**  
   - 计算 2D 功率谱与径向谱；  
   - 构造**补偿谱**和**累计径向能量曲线**，更直观地识别主导波数与能量分布；  
   - 计算空间平均的时间 PSD，分析能量主要集中在哪些频段。

4. **Step 3：对称性与各向异性**  
   - 比较沿 \(k_x\) / \(k_y\) 方向的谱切片；  
   - 定义并绘制尺度依赖的各向异性比值 \(E_{kx}(k)/E_{ky}(k)\)；  
   - 计算谱在镜像 / 90° 旋转下的相对误差；  
   - 检查前几阶 SVD 模态在 x、y 方向翻转下的相关系数。

最终在 `figures/report_composite_figure.pdf` 中给出一张 2×2 综合图，对 Q1–Q4
的主要结论做图形化总结，可直接用于 LaTeX 报告。

---

## 4. 代码结构

```text
Spatiotemporal-Data-Analysis/
├── data/                    # 数据集（未加入版本控制）
├── figures/                 # 所有输出图片
├── notebooks/               # 英文 / 中文 Jupyter notebook
├── src/                     # 分析模块
│   ├── data_loader.py       #   数据加载与基础诊断
│   ├── svd_analysis.py      #   SVD/PCA 模态分析工具
│   ├── spectral_analysis.py #   空间 / 时间谱分析函数
│   ├── symmetry_analysis.py #   对称性与各向异性指标
│   └── visualization.py     #   统一的绘图工具
├── main.py                  # 一键运行完整流程的入口脚本
├── pyproject.toml           # 依赖与打包信息
└── README*.md               # 本文件及英文说明
```

`src/` 下的模块既可以从 `main.py` 调用，也可以在 notebook 中单独使用。

---

## 5. Jupyter Notebooks

`notebooks/` 目录中提供了两个可选的 notebook：

- `notebooks/main_analysis_en.ipynb`：英文版本，包含详细英文说明；  
- `notebooks/main_analysis.ipynb`：中文版本，代码一致，注释和讲解为中文。

两份 notebook 都按照上面的四个 Step 编排，并在末尾生成与报告一致的综合图。

启动示例（中文或英文都可以）：

```bash
jupyter lab notebooks/main_analysis.ipynb
```

---

## 6. 依赖与运行方式

**基础依赖：**

- Python ≥ 3.10  
- `numpy` ≥ 1.24  
- `matplotlib` ≥ 3.7  
- `jupyter`（可选，仅在使用 notebook 时需要）

### 命令行运行完整分析

```bash
pip install .
python main.py
```

### Notebook 方式交互探索

```bash
pip install .[notebook]
jupyter lab notebooks/main_analysis.ipynb
```

所有生成的图像会被写入 `figures/` 目录，用于后续报告排版。
