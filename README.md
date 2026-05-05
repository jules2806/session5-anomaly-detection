# Session 5 — Advanced Anomaly Detection

**Course:** Machine Learning III (Unsupervised Learning) — @Albert School  
**Assessment:** Practical group project (1–3 students) | 3 hours | Graded

---

## Business Context

A major manufacturing firm operates heavy machinery that costs **€100,000 per hour** of downtime when it fails. Standard rules-based monitoring is no longer sufficient.

This project builds an **unsupervised anomaly detection pipeline** that flags potential machine failures before they occur, while minimising false alarms for the maintenance crew.

**Cost matrix:**
| Error type | Financial impact |
|------------|-----------------|
| False Positive (unnecessary alert) | €500 |
| False Negative (missed failure) | €15,000 |

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

- 10,000 observations × 14 features
- Sensor readings: air temperature, process temperature, rotational speed, torque, tool wear
- Machine type (L / M / H quality grade)
- Ground truth label: `Machine failure` (dropped during training, used only for final evaluation)

> The dataset is downloaded automatically when running the notebook — no manual download needed.

---

## Project Structure

```
session5-anomaly-detection/
├── Session5_Anomaly_Detection_Notebook.ipynb   # Main notebook (all 4 parts)
├── Session5_Assessment.md                       # Assignment instructions
└── README.md                                    # This file
```

---

## Notebook Structure

| Part | Content | Owner |
|------|---------|-------|
| **Part 1** | EDA & Cleaning — distributions, correlations, encoding, scaling | Jules |
| **Part 2** | Modeling — IsolationForest, OneClassSVM, LOF, EllipticEnvelope | Jules |
| **Part 3** | Visualizations — PCA, anomaly maps, deep dive LOF vs iForest | Gabriel |
| **Part 4** | Managerial conclusion — cost analysis, model recommendation | Gabriel |

---

## Models Used

| Model | Key hyperparameters | Why |
|-------|-------------------|-----|
| **Isolation Forest** | `n_estimators=200`, `contamination=0.034` | Global outliers via random partitioning — no distance, no distribution assumption |
| **One-Class SVM** | `nu=0.034`, `kernel=rbf`, `gamma=scale` | Non-linear decision boundary in sensor space |
| **Local Outlier Factor** | `n_neighbors=20`, `contamination=0.034` | Local density deviation — catches isolated failure clusters |
| **Elliptic Envelope** | `contamination=0.034`, `support_fraction=0.85` | Robust covariance estimation assuming Gaussian distribution |

> `contamination = 3.4%` is derived from the observed failure rate in the dataset (339 / 10,000).

---

## Installation

### Prerequisites

- Python **3.8+**
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or with conda:

```bash
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Full requirements

```
pandas>=1.3
numpy>=1.21
matplotlib>=3.4
seaborn>=0.11
scikit-learn>=1.0
jupyter>=1.0
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/jules2806/session5-anomaly-detection.git
cd session5-anomaly-detection

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch Jupyter
jupyter notebook Session5_Anomaly_Detection_Notebook.ipynb
```

Then run all cells: **Kernel → Restart & Run All**

> The dataset downloads automatically from UCI on first run (~2 seconds).

---

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Stable, complete notebook |
| `jules` | Jules's work (Parts 1 & 2) |
| `gabriel` | Gabriel's work (Parts 3 & 4) |
| `malika` | Malika's branch |

---

## Results Summary

The notebook concludes with a financial comparison of all four models, identifying which minimises total cost given the €500/€15,000 FP/FN ratio, and recommends an optimal contamination threshold for production deployment.

---

## Authors

- **Jules** — [jules2806](https://github.com/jules2806)
- **Gabriel** — [SacreZbb](https://github.com/SacreZbb)
