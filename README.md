<p align="center">
  <h1 align="center">SVMs, PCA, KPCA + LDA</h1>
  <p align="center">
    <strong>Classification Experiments on Gene Expression &amp; Image Data</strong>
  </p>
  <p align="center">
    <code>SVM</code>&ensp;·&ensp;<code>KNN</code>&ensp;·&ensp;<code>NCC</code>&ensp;·&ensp;<code>PCA</code>&ensp;·&ensp;<code>Kernel PCA</code>&ensp;·&ensp;<code>LDA</code>
  </p>
</p>

<br>

> Benchmarking classical ML classifiers combined with dimensionality reduction on two fundamentally different datasets: **MLL** (3-class leukemia gene expression) and **CIFAR-10** (10-class image recognition).

---

## 📂 Project Structure

```
src/
│
├── Utils.py                   # Shared data loading, preprocessing & helpers
├── config.yaml                # Centralised hyperparameters for both datasets
│
├── SVM_MLL.py                 # SVM with grid search on MLL
├── SVM_CIFAR.py               # SVM with grid search on CIFAR-10
├── Baseline_models.py         # KNN & NCC baselines on both datasets
├── kpca_lda_standalone.py     # KPCA → LDA pipeline (grid search, used as classifier)
├── kpca_lda_classifier.py     # KPCA → LDA → KNN or SVM pipeline
│
├── MLL.tab                    # MLL gene expression dataset
└── requirements.txt           # Python dependencies

Experiments Results/            # Pre-run output files (CSVs, plots, saved models)
```

---

## 🗂️ Datasets

<table>
  <tr>
    <th></th>
    <th>MLL</th>
    <th>CIFAR-10</th>
  </tr>
  <tr>
    <td><strong>Type</strong></td>
    <td>Gene expression (tabular)</td>
    <td>RGB images (32 × 32)</td>
  </tr>
  <tr>
    <td><strong>Classes</strong></td>
    <td>3 — ALL · AML · MLL</td>
    <td>10 — airplane, automobile, …</td>
  </tr>
  <tr>
    <td><strong>Features</strong></td>
    <td>12 533</td>
    <td>3 072 (32 × 32 × 3)</td>
  </tr>
  <tr>
    <td><strong>Samples</strong></td>
    <td>43 train / 29 test</td>
    <td>50 000 train / 10 000 test</td>
  </tr>
  <tr>
    <td><strong>Source</strong></td>
    <td><code>MLL.tab</code> (bundled)</td>
    <td>Auto-downloaded via Keras</td>
  </tr>
</table>

---

## ⚙️ Methods

### Dimensionality Reduction

| Technique | Description |
|:----------|:------------|
| **PCA** | Retains a configurable percentage of variance (default 90–95%) |
| **KPCA** | Kernel PCA with RBF or linear kernel; components tuned via grid search |
| **LDA** | Linear Discriminant Analysis applied after KPCA to maximise class separability |

### Classifiers

| Script | Classifier | Reduction |
|:-------|:-----------|:----------|
| `SVM_MLL.py` | SVM (RBF kernel) | PCA |
| `SVM_CIFAR.py` | SVM (RBF kernel) | PCA |
| `Baseline_models.py` | KNN · NCC | PCA |
| `kpca_lda_standalone.py` | LDA (max-likelihood decision) | KPCA + LDA |
| `kpca_lda_classifier.py` | KNN or SVM | KPCA + LDA |

---

## 📊 Experimental Findings

### CIFAR-10 — Image Classification

<table>
<tr><td>

#### 🔑 Key Takeaway

> The **RBF-kernel SVM** was the best overall model. Simpler **PCA + SVM (RBF)** outperformed the more complex KPCA + LDA pipelines on this dataset.

</td></tr>
</table>

**Non-linear separability & kernel selection.**
CIFAR-10 data is not linearly separable. Linear-kernel SVMs struggled severely — training times reached up to 30 hours for large C values without converging. Non-linear kernels (RBF, Polynomial) performed vastly better, with the **RBF kernel emerging as the optimal choice**.

**Regularisation matters (soft margin).**
For linear SVMs, very small C values produced better accuracy *and* drastically faster training. A small C creates a soft margin that tolerates misclassifications, preventing overfitting on noisy image data and improving generalisation.

**Baseline models failed.**
KNN and NCC performed the worst by a significant margin. NCC assumes each class clusters around a single centroid — but the "mean image" of a CIFAR-10 class is essentially noise. KNN's reliance on Euclidean pixel distance is too crude to capture the complex features needed to distinguish visually similar classes (e.g. *cat* vs *dog*).

**Scaling had negligible impact.**
MinMax [0, 1], MinMax [−1, 1], and Z-score standardisation all produced very similar accuracy. `StandardScaler` required slightly more training time, likely because unbounded values slow convergence.

**KPCA + LDA underperformed.**
LDA assumes normally distributed classes with similar variances — conditions CIFAR-10 violates due to high intra-class variance and overlapping class means. Hardware constraints also prevented an exhaustive hyperparameter search for the KPCA models.

---

### MLL — Gene Expression

<table>
<tr><td>

#### 🔑 Key Takeaway

> **KPCA + LDA** achieved the highest overall accuracy, projecting 12 533 dimensions into just 2D with near-perfect class separation. Linear models dominated.

</td></tr>
</table>

**Inherent linear separability.**
With thousands of dimensions but very few samples, MLL data is almost perfectly linearly separable. Linear SVMs and KPCA with a linear kernel achieved the best results.

**Strong regularisation is crucial.**
The best performances consistently used exceptionally small C values. The tiny dataset makes large C values cause severe overfitting and poor generalisation.

**PCA trade-offs.**
Retaining 95% of variance reduced dimensionality from 12 533 → ~35 features. This caused a slight accuracy drop but cut training time by an **order of magnitude**.

**Baseline models excelled.**
Unlike on CIFAR-10, KNN and NCC (with PCA) achieved **> 86% accuracy**, confirming that the three leukemia types form highly distinct, well-separated clusters in gene-expression space.

**KPCA + LDA was exceptional.**
Linear KPCA + LDA projected the high-dimensional data into a 2D space where the classes were almost flawlessly separated — a simple linear classifier could easily distinguish them.

---

### Side-by-Side Comparison

| Aspect | CIFAR-10 | MLL |
|:-------|:---------|:----|
| **Best pipeline** | PCA + SVM (RBF) | KPCA + LDA |
| **Separability** | Non-linear | Nearly linear |
| **Regularisation** | Small C preferred | Small C crucial |
| **Baseline models** | Failed | Excelled (> 86%) |
| **KPCA + LDA** | Underperformed | Best overall |
| **Scaling impact** | Negligible | — |

---

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

> [!NOTE]
> TensorFlow / Keras is required only to download CIFAR-10. If you are only using the MLL dataset you can skip it.

### Running the Experiments

All scripts must be run from inside `src/` — `MLL.tab` is loaded via a relative path:

```bash
cd src/
```

<details>
<summary><strong>SVM on MLL</strong></summary>

```bash
python SVM_MLL.py
```

Runs a grid search over `C` and `gamma` for an RBF SVM across three scalers (MinMaxScaler, MinMaxScaler [−1, 1], StandardScaler) with PCA preprocessing.
</details>

<details>
<summary><strong>SVM on CIFAR-10</strong></summary>

```bash
python SVM_CIFAR.py
```

Same pipeline as above, but on a configurable subsample of CIFAR-10 (default 20 000 training images).
</details>

<details>
<summary><strong>KNN & NCC baselines</strong></summary>

```bash
python Baseline_models.py
```

Runs a KNN grid search (over `k`) and a Nearest Centroid classifier. Switch between datasets by changing `dataset_name` at the top of `main()`.
</details>

<details>
<summary><strong>KPCA + LDA standalone</strong></summary>

```bash
python kpca_lda_standalone.py
```

Grid searches KPCA hyperparameters (components, kernel, gamma) and uses the resulting LDA projection directly for classification.
</details>

<details>
<summary><strong>KPCA + LDA + downstream classifier</strong></summary>

```bash
python kpca_lda_classifier.py
```

Fits KPCA → LDA → KNN or SVM in a single pipeline. Select the classifier by setting `model_name` to `"KPCA_LDA_KNN"` or `"KPCA_LDA_SVM"` at the top of `main()`.
</details>

---

## 🛠️ Configuration

All hyperparameters live in **`config.yaml`** under two top-level keys — `MLL` and `CIFAR10`. Read at runtime by `Utils.load_config(dataset_name)`, it is the single source of truth for grid search bounds, PCA thresholds, CV folds, sample sizes, and KPCA settings.

Each script's `main()` also exposes a few runtime flags:

| Flag | Description |
|:-----|:------------|
| `dataset_name` | `'MLL'` or `'CIFAR-10'` |
| `PCA_enable` | Toggle PCA on / off |
| `Multiple_cores_enable` | Use all CPU cores for grid search |
| `GrayScale` | Convert CIFAR-10 to grayscale (CIFAR scripts only) |
| `model_name` | `"KPCA_LDA_KNN"` or `"KPCA_LDA_SVM"` (KPCA classifier script only) |

---

## 📁 Output Files

Each run creates a subdirectory next to the script (e.g. `MLL_rbf_MinMaxScaler_PCA_output_files/`):

| File | Contents |
|:-----|:---------|
| `results_*.csv` | Full grid search CV results, ranked by score |
| `best_est_results_*.csv` | Best model's parameters & evaluation metrics |
| `conf_matrix_*.png` | Confusion matrix heatmap on the test set |
| `contour_grid_*.png` | C vs γ accuracy contour (SVM with RBF / poly kernel) |
| `*_best_model.pkl` | Saved best model (joblib) |
| `classification_example_*.png` | Example misclassified image (CIFAR-10 only) |
