# SVMs, PCA, KPCA + LDA — Classification Experiments

This project benchmarks classical machine learning classifiers combined with dimensionality reduction techniques on two datasets: **MLL** (gene expression, 3-class leukemia) and **CIFAR-10** (10-class image recognition).

---

## Project Structure

```
src/
├── Utils.py                   # Shared data loading, preprocessing, and helper functions
├── config.yaml                # Centralised hyperparameters for both datasets
├── SVM_MLL.py                 # SVM with grid search on the MLL dataset
├── SVM_CIFAR.py               # SVM with grid search on CIFAR-10
├── Baseline_models.py         # KNN and NCC baselines on both datasets
├── kpca_lda_standalone.py     # KPCA → LDA pipeline (grid search, used as classifier)
├── kpca_lda_classifier.py     # KPCA → LDA → KNN or SVM pipeline
├── MLL.tab                    # MLL gene expression dataset
└── requirements.txt           # Python dependencies

Experiments Results/           # Pre-run output files (CSVs, plots, saved models)
```

---

## Datasets

| Dataset | Type | Classes | Samples | Notes |
|---|---|---|---|---|
| **MLL** | Gene expression (tabular) | 3 — ALL, AML, MLL | 43 train / 29 test | 12,533 features; loaded from `MLL.tab` |
| **CIFAR-10** | RGB images (32×32) | 10 — airplane, automobile, … | 50,000 train / 10,000 test | Downloaded automatically via Keras on first run |

---

## Methods

### Dimensionality Reduction

- **PCA** — retains a configurable percentage of variance (default 90–95%).
- **KPCA** — kernel PCA with RBF or linear kernel; number of components tuned via grid search.
- **LDA** — Linear Discriminant Analysis, applied after KPCA to maximise class separability.

### Classifiers

| Script | Classifier | Reduction |
|---|---|---|
| `SVM_MLL.py` | SVM (RBF kernel) | PCA |
| `SVM_CIFAR.py` | SVM (RBF kernel) | PCA |
| `Baseline_models.py` | KNN, NCC | PCA |
| `kpca_lda_standalone.py` | LDA itself (max-likelihood decision) | KPCA + LDA |
| `kpca_lda_classifier.py` | KNN or SVM on top of LDA space | KPCA + LDA |

---

## Experimental Findings

### CIFAR-10 (Image Classification)

CIFAR-10 is characterised by high noise and inherent non-linearity. The key observations are summarised below.

**Non-linear separability and kernel selection.** The data is not linearly separable. Linear-kernel SVMs struggled significantly, with training times reaching up to 30 hours for large penalty values of C without converging. Non-linear kernels (RBF and Polynomial) yielded vastly superior performance, and the **RBF kernel emerged as the best overall model** for this dataset.

**The role of regularisation (soft margin).** For linear SVMs, very small C values produced better accuracy and drastically faster training. A small C acts as a regulariser, creating a soft margin that tolerates misclassifications and prevents overfitting on noisy image data, thereby improving generalisation.

**Failure of baseline models.** KNN and NCC performed the worst by a significant margin. NCC assumes that each class clusters around a single centroid, but the "mean image" of a CIFAR-10 class is essentially noise. KNN relies on Euclidean distance between raw pixel values, which is a poor metric for capturing the complex features needed to distinguish visually similar classes (e.g. "cat" vs "dog").

**Impact of scaling methods.** The choice of scaler (MinMax [0,1], MinMax [−1,1], or Z-score standardisation) had a negligible impact on accuracy. However, `StandardScaler` (Z-score) required slightly more training time, likely because unbounded feature values slow algorithmic convergence.

**Limitations of the KPCA + LDA pipeline.** The simpler PCA + SVM (RBF) pipeline outperformed the more complex KPCA + LDA + Classifier pipelines. LDA assumes normally distributed classes with similar variances — conditions that CIFAR-10 does not meet due to high intra-class variance and class means that are too close together. Hardware limitations also prevented an exhaustive hyperparameter search for the KPCA models.

### MLL (Gene Expression)

The MLL dataset presents the opposite challenge: extremely high dimensionality (12,533 features) but very few samples (72 total across 3 classes).

**Inherent linear separability.** With thousands of dimensions and very few data points, the MLL data is almost perfectly linearly separable. Models relying exclusively on linear components (Linear SVMs, KPCA with a linear kernel) achieved the highest performance.

**Crucial need for strong regularisation.** The best results were consistently obtained with exceptionally small C values. Because the dataset is so small, large C values lead to severe overfitting and poor generalisation.

**Trade-offs of using PCA.** Retaining 95% of variance reduced dimensionality from 12,533 to roughly 35 features. This caused a slight drop in accuracy — some class discriminability is inevitably lost — but decreased training time by an order of magnitude.

**High efficacy of baseline models.** In stark contrast to CIFAR-10, KNN and NCC (with PCA) achieved over 86% accuracy on MLL, indicating that the three leukemia types form highly distinct, well-separated clusters in the gene-expression feature space.

**Exceptional performance of KPCA + LDA.** The KPCA + LDA pipeline proved to be the best feature-extraction method for this dataset, yielding the **highest overall accuracy**. Linear variants of these models projected the 12,533-dimensional data into a 2-dimensional space where the classes were almost flawlessly separated, allowing a simple linear classifier to distinguish them easily.

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow/Keras is required only to download CIFAR-10. If you are only using the MLL dataset you can skip it.

---

## Running the Experiments

All scripts must be run from inside the `src/` directory — `MLL.tab` is loaded via a relative path and the scripts will fail if invoked from elsewhere:

```bash
cd src/
```

### SVM on MLL

```bash
python SVM_MLL.py
```

Runs a grid search over `C` and `gamma` for an RBF SVM across three scalers (MinMaxScaler, MinMaxScaler [−1,1], StandardScaler) with PCA preprocessing.

### SVM on CIFAR-10

```bash
python SVM_CIFAR.py
```

Same pipeline as above, but on a configurable subsample of CIFAR-10 (default 20,000 training images).

### KNN and NCC baselines

```bash
python Baseline_models.py
```

Runs a KNN grid search (over `k`) and a Nearest Centroid classifier. Switch between datasets by changing `dataset_name` at the top of `main()`.

### KPCA + LDA standalone

```bash
python kpca_lda_standalone.py
```

Grid searches KPCA hyperparameters (number of components, kernel, gamma) and uses the resulting LDA projection directly for classification.

### KPCA + LDA + downstream classifier

```bash
python kpca_lda_classifier.py
```

Fits KPCA → LDA → KNN or SVM in a single pipeline. Select the classifier by setting `model_name` to `"KPCA_LDA_KNN"` or `"KPCA_LDA_SVM"` at the top of `main()`.

---

## Configuration

All experiment hyperparameters are centralised in **`config.yaml`** under two top-level keys — `MLL` and `CIFAR10`. This file is read at runtime by `Utils.load_config(dataset_name)` and is the single source of truth for grid search bounds, PCA thresholds, CV folds, sample sizes, and KPCA settings.

Each script's `main()` still exposes a few runtime flags:

| Flag | Description |
|---|---|
| `dataset_name` | `'MLL'` or `'CIFAR-10'` |
| `PCA_enable` | Toggle PCA on/off |
| `Multiple_cores_enable` | Use all CPU cores for grid search |
| `GrayScale` | Convert CIFAR-10 to grayscale before training (CIFAR scripts only) |
| `model_name` | `"KPCA_LDA_KNN"` or `"KPCA_LDA_SVM"` (KPCA classifier script only) |

---

## Output Files

Each experiment run creates a subdirectory next to the script (e.g. `MLL_rbf_MinMaxScaler_PCA_output_files/`) containing:

| File | Contents |
|---|---|
| `results_*.csv` | Full grid search CV results, ranked by score |
| `best_est_results_*.csv` | Best model's parameters and evaluation metrics |
| `conf_matrix_*.png` | Confusion matrix heatmap on the test set |
| `contour_grid_*.png` | C vs gamma accuracy contour (SVM with RBF/poly kernel) |
| `*_best_model.pkl` | Saved best model (joblib) |
| `classification_example_*.png` | Example misclassified image (CIFAR-10 only) |