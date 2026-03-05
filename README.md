# ME5311 Project 1 — Spatio‑temporal data analysis

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README_EN.md)
[![cn](https://img.shields.io/badge/lang-中文-red.svg)](README.md)

**Author**: Tan Kailai (e1554333@u.nus.edu)  
**Module**: NUS ME5311 — Data‑driven modelling and control

This repository contains my **personal implementation** of the ME5311 Project 1
analysis pipeline. The focus is on *learning from the given dataset itself*:
no dynamical model is built and no prediction is attempted — everything is
based on statistics and structures already present in the simulation output.

---

## 1. Assignment context

The official brief for Project 1 asks us to use data‑driven tools to answer, at
least qualitatively, the following questions:

1. **Q1 – Dominant spatial structures**  
   What large‑scale patterns can be extracted from the field (e.g. via SVD)?
2. **Q2 – Energy across spatial scales**  
   How is variance/energy distributed over different wavenumbers?
3. **Q3 – Evidence of external periodic forcing**  
   Can a preferred forcing wavenumber be inferred directly from the spectrum?
4. **Q4 – Symmetry and anisotropy**  
   Do the spatial/temporal statistics reflect isotropy, or clear directional
   preferences?

The code in this repository is organised explicitly around these questions.

---

## 2. Dataset

The project uses a single, relatively large, spatio‑temporal dataset:

| Property      | Value                                  |
|---------------|----------------------------------------|
| Shape         | `(15000, 64, 64, 2)`                  |
| Description   | Two‑component vector field on a 2‑D periodic domain |
| Time step     | `Δt = 0.2`                            |
| Total duration| `3000` simulation time units          |
| File          | `data/vector_64.npy`                  |

Each snapshot records both velocity components on a `64×64` Cartesian grid.  
The raw array is the **only input** to all subsequent analysis.

---

## 3. Analysis workflow

My workflow is split into four stages, which roughly parallel the lecture
content:

1. **Pre‑processing & basic statistics (Step 0)**  
   - Load the vector field and separate it into mean flow and fluctuations.  
   - Form vorticity and divergence fields.  
   - Inspect representative snapshots and global kinetic‑energy time series.  
   - Plot empirical PDFs of vorticity/divergence to gauge non‑Gaussianity.

2. **Low‑rank SVD of the fluctuation field (Step 1)**  
   - Compute an economy‑size SVD of the snapshot matrix.  
   - Examine the singular spectrum, cumulative captured energy, and a
     **reconstruction‑error curve** as a function of retained modes.  
   - Visualise the leading spatial modes and their time coefficients.

3. **Spectral characterisation in space and time (Step 2)**  
   - Build 2‑D power spectra in wavenumber space and their radial averages.  
   - Construct **compensated spectra** and **cumulative radial energy curves**
     to highlight dominant length scales and potential power‑law ranges.  
   - Compute spatially averaged temporal PSDs and locate energetic frequency
     bands.

4. **Symmetry and anisotropy diagnostics (Step 3)**  
   - Compare spectral cuts along \(k_x\) and \(k_y\).  
   - Quantify a **scale‑dependent anisotropy ratio** \(E_{kx}(k)/E_{ky}(k)\).  
   - Evaluate simple mirror/rotation symmetry errors of the 2‑D spectrum.  
   - Inspect symmetry properties of the leading SVD modes.

A single composite figure, saved as `figures/report_composite_figure.pdf`,
summarises key results for Q1–Q4 and is intended to be used directly in the
written report.

---

## 4. Repository layout

```text
Spatiotemporal-Data-Analysis/
├── data/                    # dataset (ignored in version control)
├── figures/                 # all output figures
├── notebooks/               # optional Jupyter notebooks (EN & CN)
├── src/                     # analysis modules
│   ├── data_loader.py       #   data loading & basic diagnostics
│   ├── svd_analysis.py      #   SVD/PCA analysis utilities
│   ├── spectral_analysis.py #   spatial & temporal spectral tools
│   ├── symmetry_analysis.py #   symmetry & anisotropy indicators
│   └── visualization.py     #   centralised plotting helpers
├── main.py                  # single entry point for the full workflow
├── pyproject.toml           # dependencies & packaging metadata
└── README*.md               # this file and the Chinese version
```

The scripts under `src/` are written to be reusable from both the command line
and the notebooks.

---

## 5. Notebooks

Two interactive notebooks live in `notebooks/`:

- `notebooks/main_analysis_en.ipynb` — English narrative of the full pipeline.  
- `notebooks/main_analysis.ipynb` — Same code path, commentary in Chinese.

Both notebooks follow the four stages above and ultimately regenerate the
composite figure used in the report. They are optional; all analysis can also
be run from `main.py`.

To open the English notebook:

```bash
jupyter lab notebooks/main_analysis_en.ipynb
```

---

## 6. Dependencies and usage

Minimal requirements:

- Python ≥ 3.10  
- `numpy` ≥ 1.24  
- `matplotlib` ≥ 3.7  
- `jupyter` (optional, for notebook use)

Install the package in editable mode and run the full pipeline:

```bash
pip install .
python main.py
```

For a notebook‑centric workflow:

```bash
pip install .[notebook]
jupyter lab notebooks/main_analysis_en.ipynb
```

All generated figures are written to the `figures/` directory.
