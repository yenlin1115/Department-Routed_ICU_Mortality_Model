# Department-Routed ICU Mortality Model — PhysioNet 2012 Challenge

## Overview

This repository contains the implementation of a two-stage department-routed model
for in-hospital mortality prediction in ICU patients, evaluated on the PhysioNet
2012 Challenge dataset (Sets A, B, and C).

**Architecture:**
- **Stage 1 — GBM backbone:** `HistGradientBoostingClassifier` trained on all Set A
  patients using 187 conditioned physiological features (no ICU type information).
- **Stage 2 — Department routing GLM:** Logistic regression on
  `[logit(backbone_prob), ICU_CCU, ICU_CSRU, ICU_MICU, ICU_SICU]`.
  ICU one-hot indicators allow department-specific risk calibration adjustments
  (partially pooled design).

**Evaluation metrics:**
- **Event 1 (E1):** min(Sensitivity, PPV) at the optimal threshold (tuned on Set A only) — higher is better.
- **Event 2 (E2):** Hosmer-Lemeshow calibration statistic normalised by risk-score range — lower is better.

**Results:**

| Model                     | B E1   | B E2  | C E1   | C E2  |
|---------------------------|--------|-------|--------|-------|
| Citi L, Barbieri R. model | 0.5199 | 13.53 | 0.5276 | 21.20 |
| Dept-routed               | 0.5216 | 11.85 | 0.5197 | 23.84 |
| Target                    | >0.5200| <13.54| >0.5345| <17.88|


The department-routed model beats the Citi L, Barbieri R.  on both Set B metrics.

---

## Requirements

- Python 3.9 or higher
- GCC (for compiling the PhysioNet score binary — included as `score.c`)

Install Python dependencies:

If use python 3(Mac):

```bash
pip3 install -r requirements.txt
```
If use python:
```bash
pip install -r requirements.txt
```
---

## Data files (included in `data/`)

| File | Description |
|------|-------------|
| `set-a_features_0257.npz` | Pre-extracted 187-dim features, Set A (4,000 patients) |
| `set-b_features_0257.npz` | Pre-extracted 187-dim features, Set B (4,000 patients) |
| `set-c_features_0257.npz` | Pre-extracted 187-dim features, Set C (4,000 patients) |
| `Outcomes-a.txt` | Ground-truth labels, Set A |
| `Outcomes-b.txt` | Ground-truth labels, Set B |
| `Outcomes-c.txt` | Ground-truth labels, Set C |
| `score.c` | Official PhysioNet scoring program (C source) |
| `lm_feat_mis0.mat` | Citi & Barbieri MATLAB SVM model (Citi L, Barbieri R. , for comparison curves) |
| `lm_svm_scores.npz` | Pre-computed SVM scores from MATLAB model (speeds up Citi L, Barbieri R.  evaluation) |

---

## How to run

If use Python 3(Mac):
```bash
python3 department_routed_model.py
```
If use Python:
```bash
python department_routed_model.py
```

This will:
1. Compile `score.c` automatically (requires GCC)
2. Train the GBM backbone on Set A
3. Train the department routing GLM on Set A
4. Evaluate on Sets B and C using the official PhysioNet scorer
5. Print a results table (E1, E2 per set, AUC per set, per-ICU breakdown)
6. Save three figures to `figures/`:
   - `figure_roc_curves.png`
   - `figure_calibration_curves.png`
   - `figure_dca_curves.png`

**Optional arguments:**

```
--data_dir   Path to data directory (default: ./data)
--out_dir    Path to figures output directory (default: ./figures)
```

---

## Reproducibility notes

- All random seeds are fixed (`random_state=42`).
- The backbone is trained **only** on Set A. Sets B and C are never seen during training or threshold tuning.
- The decision threshold for Event 1 is tuned on Set A only, then applied to B and C.
- `score.c` is the official unmodified PhysioNet 2012 scoring program.
