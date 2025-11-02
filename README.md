Here’s a **ready-to-drop-in `README.md`** you can put at the repo root. It matches the code/pipeline we built together (diagnosis embeddings → clustering → DXC features → WOE/IV selection → correlation pruning → tree models on a log target → back-transform to original scale), and it includes exact setup & run steps plus submission formatting.

---

# HiLabs Risk Scoring — End-to-End Pipeline

## 1) Overview

This repository predicts **patient-level risk scores** from five CSVs:

* `patient.csv`
* `risk.csv` (contains the label `risk_score` per patient)
* `visit.csv`
* `care.csv`
* `diagnosis.csv`

**High-level approach**

1. **Load & unify** the five tables and build a patient-level feature matrix.
2. **Text → features**: turn free-text diagnosis into **semantic clusters** using **pre-trained embeddings** (SBERT or GloVe), then create **per-patient cluster counts** (`DXC_*`).
3. **Feature selection**:

   * **WOE/IV** ranking (derive a binary label from original risk via median split) → drop the **10 lowest IV** features.
   * **Correlation pruning** among numeric features (|r| ≥ 0.95), keeping the higher-IV member of each highly-correlated pair.
4. **Modeling**: train tree models (**LightGBM, XGBoost, CatBoost, RandomForest**) on **`log_risk_score`**, predict on the log scale, then **back-transform** with `exp()` to original scale for metrics and the final CSV.
5. **Output** `Prediction.csv` = (`patient_id`, `predicted_risk_score`).

This design maximizes accuracy while staying interpretable and reproducible, and avoids common pitfalls like target leakage.

---

## 2) Data Inputs & Schema Assumptions

We expect the following minimal columns (extra columns are fine):

* **patient.csv**: `patient_id`, demographics (e.g., `age`, `sex`, etc.)
* **risk.csv**: `patient_id`, `risk_score` (continuous)
* **visit.csv**: `patient_id`, `visit_type`, `facility_id`, `start_date`, `end_date`, …
* **care.csv**: `patient_id`, care events (counts/flags), timestamps
* **diagnosis.csv**: `patient_id`, **`prncpl_diag_nm`** (primary diagnosis text), codes if available

All joins are done on `patient_id`. Timestamps are parsed to create simple utilization features (counts, recency, lengths of stay), and diagnosis text is normalized and embedded.

---

## 3) Pipeline (Details)

### 3.1 Text Embedding & Clustering (Diagnosis → DXC features)

* **Normalize** `prncpl_diag_nm`: lowercasing, remove punctuation/filler words (e.g., “unspecified”, “initial encounter”), drop laterality (`left/right`), simple de-plurals, collapse spaces.
* **Embed** unique normalized strings (`dx_vocab`) using **pre-trained** models:

  * **Default**: `sentence-transformers/all-MiniLM-L6-v2` (SBERT), or
  * **Alternative**: **GloVe** (`glove-wiki-gigaword-100`) with **IDF-weighted average** of word vectors.
* **Cluster** embeddings to group semantically similar diagnoses:

  * **Agglomerative** on cosine distances (default `distance_threshold=0.20`, `linkage="average"`).
  * Optional: **HDBSCAN** for density-based clustering (labels `-1` = noise).
* **Engineer features**: count clusters per patient → wide table of `DXC_<cluster_id>` counts. Optionally keep **top-K** clusters by frequency (default `K=50`) to control feature width.

### 3.2 Core Tabular Features

* Demographics (age, sex, etc.).
* Utilization (counts of visits by type, length of stay aggregates, recency features).
* Care events/features (counts, flags).
* **Diagnosis clusters** (`DXC_*`) from the step above.

All features align to **one row per patient**. Index is `patient_id`.

### 3.3 Target Definition

* Train models on **`log_risk_score = log(risk_score)`** for stability.
* Evaluate & output on **original scale** via `np.exp(pred_log)`.

> If you used `log1p`, swap to `np.expm1` for back-transform.

### 3.4 Feature Selection

1. **WOE/IV** (against a **binary** label derived from original risk at the **median**):

   * Numeric features: quantile binning (≤10 bins).
   * Categorical features: top-50 categories + `__OTHER__`.
   * Drop the **10 lowest IV** features.
2. **Correlation pruning** (numeric only): Pearson |r| ≥ **0.95**; keep the **higher-IV** feature, drop the other.

### 3.5 Modeling

We train and compare four regressors:

* **LightGBM**: `objective='regression'`, `n_estimators=500`, `learning_rate=0.05`
* **XGBoost**: `reg:squarederror`, similar capacity (`n_estimators=500`, `max_depth=6`, `subsample=0.9`, `colsample_bytree=0.9`)
* **CatBoost**: `iterations=500`, `learning_rate=0.05`, `loss_function='RMSE'`
* **RandomForest**: `n_estimators=500`

**Evaluation metrics** (on original scale): **MAE**, **RMSE**, **R²**.

> We also provide a leakage-safe alignment by `patient_id` and robust imputation (median) before training.

---

## 4) Reproducibility & Code Quality

* **Fixed random seeds** (`random_state=42`) in splits, models, and clustering where possible.
* **Modular notebook** with clear cells for: load → preprocess → embed+cluster → features → WOE/IV → correlation prune → models → evaluation → inference.
* **No target leakage**: feature selection (IV & correlation) is applied on the **training fold** during CV in the optional pipeline version; for the single train/test split we align `X`/`y` and avoid peeking at test.

---

## 5) Repository Structure

```
.
├── README.md
├── requirements.txt
├── env_setup.sh               # creates/activates a Python env & installs deps
├── /notebooks
│   └── hilabs.ipynb           # main notebook (accepts 5 CSVs, writes Prediction.csv)
├── /src
│   ├── features.py            # (optional) helpers for text norm, embedding, clustering
│   ├── selection.py           # (optional) WOE/IV + corr-pruning utilities
│   └── train.py               # (optional) scripted runner mirroring the notebook
└── /data                      # you place patient.csv, risk.csv, visit.csv, care.csv, diagnosis.csv here
```

> The notebook alone is sufficient per the deliverables; the `/src` scripts are optional if you prefer a CLI.

---

## 6) Setup

### 6.1 Quick environment (bash)

Create `env_setup.sh` at repo root:

```bash
#!/usr/bin/env bash
set -e

# Optional: force CPU to avoid CUDA device mismatches
# export CUDA_VISIBLE_DEVICES=""

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready. Activate with: source .venv/bin/activate"
```

Make it executable:

```bash
chmod +x env_setup.sh
./env_setup.sh
```

### 6.2 `requirements.txt`

```text
pandas>=2.1
numpy>=1.26
scikit-learn>=1.4
lightgbm>=4.1
xgboost>=2.0
catboost>=1.2
sentence-transformers>=3.0
hdbscan>=0.8.33
gensim>=4.3
matplotlib>=3.8
tqdm>=4.66
```

> If you only use GloVe (no SBERT), you can drop `sentence-transformers` and `hdbscan`.

---

## 7) How to Run

### 7.1 Notebook (required deliverable)

1. Place the five CSVs in `./data/`:

```
./data/patient.csv
./data/risk.csv
./data/visit.csv
./data/care.csv
./data/diagnosis.csv
```

2. Launch Jupyter and open the notebook:

```bash
source .venv/bin/activate
jupyter lab  # or jupyter notebook
```

3. In `/notebooks/hilabs.ipynb`, set the input paths (top cell).
4. Run all cells.
5. The notebook writes:

```
./Prediction.csv   # columns: patient_id, predicted_risk_score
```

and prints model metrics (MAE / RMSE / R²) on the original risk scale.

### 7.2 Script (optional)

If you mirror the notebook in `/src/train.py`, you can do:

```bash
python -m src.train \
  --patient ./data/patient.csv \
  --risk ./data/risk.csv \
  --visit ./data/visit.csv \
  --care ./data/care.csv \
  --diagnosis ./data/diagnosis.csv \
  --out ./Prediction.csv
```

---

## 8) Feature Selection Logic & Assumptions

* **Text → clusters**: semantically similar diagnoses (e.g., *“pain in left leg”* vs *“pain in lower leg”*) should map to the same/similar clusters; we count per-patient occurrences (DXC_*).
* **IV ranking**: convert the continuous risk to a **binary label** at the **median** of original risk (not log). This is a robust, label-agnostic way to score predictive signal across features. We **drop exactly 10** lowest-IV features.
* **Correlation pruning**: for numeric features with Pearson |r| ≥ **0.95**, keep the **higher-IV** one. This removes redundancy while preserving predictive capacity.
* **Log target**: training on `log_risk_score` stabilizes distributions; predictions are back-transformed with `exp()`.
* **Imputation**: median imputation on train; apply same to test.

> Thresholds (`K=50` top DX clusters, IV drop=10, corr threshold=0.95) are **configurable** in the notebook top-cells.

---

## 9) Model Architecture & Tuning

* **Tree models** are robust to mixed feature scales, sparse `DXC_*` counts, and non-linearities:

  * **LightGBM**: `n_estimators=500`, `learning_rate=0.05` (fast, accurate baseline).
  * **XGBoost**: `max_depth=6`, `subsample=0.9`, `colsample_bytree=0.9`.
  * **CatBoost**: strong baseline on tabular even with default settings.
  * **RandomForest**: provides a stable, low-variance reference.
* You can add CV and hyper-param search (e.g., Optuna or sklearn’s `RandomizedSearchCV`). We include leakage-safe alignment by `patient_id`; for group/time splits, swap `train_test_split` with `GroupKFold` or a time-based split.

---

## 10) Interpretability & Explainability

* **WOE/IV** ranking (global): communicates directional signal and feature value separation wrt high/low risk.
* **Model importances** (LightGBM/XGB/Cat): show which features drive predictions after selection.
* **(Optional)** SHAP for per-feature contribution analysis; omitted in default run to keep runtime lightweight, but helper code is included and can be enabled.

---

## 11) Output & Submission

* The notebook writes `Prediction.csv` with **exact columns**:

```
patient_id,predicted_risk_score
<id1>,<float>
<id2>,<float>
...
```

* Rename to the required format before submitting:

```
TeamName_HiLabs_Risk_Score.csv
```

* Submit the **public GitHub link** and the **CSV**.

---

## 12) Troubleshooting

* **CUDA device mismatch** (e.g., SBERT on GPU, tensors on CPU):

  * Force CPU with `export CUDA_VISIBLE_DEVICES=""` *or* pass `device="cpu"` to `SentenceTransformer`.
* **KeyError on alignment**: make sure both `X` and `y` are indexed by `patient_id`; we intersect indices before training.
* **Too many DXC columns**: reduce `K` (top-K clusters) or raise `distance_threshold` to merge clusters.
* **Suspiciously high R²**: ensure **no leakage** (do not compute IV or correlation using test). We align correctly and avoid peeking; for full rigor, wrap selection into a CV pipeline.

---

## 13) Attribution

* **Sentence embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (or GloVe via `gensim`).
* Libraries: **pandas**, **numpy**, **scikit-learn**, **lightgbm**, **xgboost**, **catboost**, **gensim**, **hdbscan** (optional).

---

## 14) Re-running With Different Settings

Edit the notebook top-cell parameters:

```python
DX_TOPK = 50          # number of diagnosis clusters to keep
AGGLO_DIST_THR = 0.35 # clustering tightness (cosine distance)
IV_DROP_N = 15        # drop bottom N by IV
CORR_THR = 0.7       # correlation pruning threshold
```

Re-run all cells; the pipeline will regenerate features, select/prune, retrain, and rewrite `Prediction.csv`.

---

**Contact / Notes**: If reviewers need a CLI run or Dockerfile, we can add a `Dockerfile` and `make predict` target; the notebook remains the single source of truth for the deliverables.
