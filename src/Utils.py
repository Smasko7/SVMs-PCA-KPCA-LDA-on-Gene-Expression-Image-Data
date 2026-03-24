"""Shared utilities: data loading, preprocessing, visualisation, and experiment helpers."""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

import yaml

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.datasets import cifar10
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tabulate import tabulate

log = logging.getLogger(__name__)

RANDOM_STATE = 42

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(dataset_name: str) -> dict:
    """Return the config block for *dataset_name* from config.yaml."""
    with _CONFIG_PATH.open() as fh:
        full = yaml.safe_load(fh)
    key = "CIFAR10" if dataset_name == "CIFAR-10" else dataset_name
    return full[key]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cifar10(sample_size: int | None = None) -> tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    total_samples = X_train.shape[0]
    log.info("Original Training Data Shape: %s", X_train.shape)
    log.info("Number of total images: %d", total_samples)

    if sample_size and sample_size < total_samples:
        log.info("Keeping only %d train samples to reduce training time", sample_size)
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=sample_size,
            stratify=y_train, random_state=RANDOM_STATE,
        )

    return X_train, y_train, X_test, y_test


def load_mll() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv('MLL.tab', sep='\t')
    df = df.drop(index=[0, 1]).reset_index(drop=True)

    label_col = df.columns[0]
    y_raw = df[label_col]
    X = df.drop(columns=[label_col]).to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    log.info("Classes: %s", le.classes_)
    log.info("Data shape: %s", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, stratify=y, random_state=RANDOM_STATE,
    )
    return X_train, y_train, X_test, y_test


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _apply_scaler_and_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_name: str | None,
    pca_info_threshold: float,
    pca_enable: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Scale data then optionally reduce dimensionality with PCA."""
    if scaler_name == "StandardScaler":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    elif scaler_name == "MinMaxScaler2":  # scale to [-1, 1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        # Uniform normalization to [0, 1] (pixel values assumed in [0, 255])
        X_train_scaled = X_train.astype('float32') / 255.0
        X_test_scaled = X_test.astype('float32') / 255.0

    if pca_enable:
        pca = PCA(n_components=pca_info_threshold, svd_solver='full')
        X_train_out = pca.fit_transform(X_train_scaled)
        X_test_out = pca.transform(X_test_scaled)
        n_components = pca.n_components_
    else:
        X_train_out = X_train_scaled
        X_test_out = X_test_scaled
        n_components = X_train.shape[1]

    return X_train_out, X_test_out, n_components


def preprocess_cifar10(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_name: str | None = None,
    pca_info_threshold: float = 0.9,
    pca_enable: bool = True,
    grayscale: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    if grayscale:
        # Luminosity method coefficients for RGB → grayscale
        X_train = np.dot(X_train, [0.2989, 0.5870, 0.1140])
        X_test = np.dot(X_test, [0.2989, 0.5870, 0.1140])

    # Flatten images to 1-D feature vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return _apply_scaler_and_pca(X_train, X_test, scaler_name, pca_info_threshold, pca_enable)


def preprocess_mll(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_name: str | None = None,
    pca_info_threshold: float = 0.95,
    pca_enable: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    return _apply_scaler_and_pca(X_train, X_test, scaler_name, pca_info_threshold, pca_enable)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_contour_from_grid(
    run_id: str,
    grid: GridSearchCV,
    score_type: str = "mean_test_score",
) -> str:
    results = grid.cv_results_
    parameters = results["params"]
    kernel = results["param_kernel"]

    if kernel[0] == 'poly':
        best = grid.best_params_
        fix_degree = best["degree"]
        fix_coef0 = best["coef0"]
        fix_C = best["C"]
        fix_gamma = best["gamma"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: C vs gamma (degree & coef0 fixed at best values)
        df1 = pd.DataFrame([
            {"C": p["C"], "gamma": p["gamma"], "score": results[score_type][i]}
            for i, p in enumerate(parameters)
            if p["degree"] == fix_degree and p["coef0"] == fix_coef0
        ]).sort_values(["C", "gamma"])

        C_vals = sorted(df1["C"].unique())
        gamma_vals = sorted(df1["gamma"].unique())
        if len(C_vals) < 2 or len(gamma_vals) < 2:
            axes[0].set_title("Cannot make contour plot for C and gamma")
        else:
            cp1 = axes[0].contourf(gamma_vals, C_vals,
                                   df1["score"].values.reshape(len(C_vals), len(gamma_vals)),
                                   levels=20)
            axes[0].set(xscale="log", yscale="log", xlabel="gamma", ylabel="C",
                        title=f"C vs gamma (deg={fix_degree}, coef0={fix_coef0})")
            fig.colorbar(cp1, ax=axes[0])

        # Right: coef0 vs degree (C & gamma fixed at best values)
        df2 = pd.DataFrame([
            {"coef0": p["coef0"], "degree": p["degree"], "score": results[score_type][i]}
            for i, p in enumerate(parameters)
            if p["C"] == fix_C and p["gamma"] == fix_gamma
        ]).sort_values(["coef0", "degree"])

        coef0_vals = sorted(df2["coef0"].unique())
        degree_vals = sorted(df2["degree"].unique())
        if len(coef0_vals) < 2 or len(degree_vals) < 2:
            axes[1].set_title("Cannot make contour plot for coef0 and degree")
        else:
            cp2 = axes[1].contourf(degree_vals, coef0_vals,
                                   df2["score"].values.reshape(len(coef0_vals), len(degree_vals)),
                                   levels=20)
            axes[1].set(xlabel="degree", ylabel="coef0",
                        title=f"coef0 vs degree (C={fix_C}, gamma={fix_gamma})")
            fig.colorbar(cp2, ax=axes[1])

        fig.suptitle(f"Poly SVM Contours – {score_type}", fontsize=14)
        fname = f"poly_dual_contours_{run_id}.png"

    else:
        df = pd.DataFrame({
            "C": [p["C"] for p in parameters],
            "gamma": [p["gamma"] for p in parameters],
            "score": results[score_type],
        }).sort_values(["C", "gamma"])

        C_vals = sorted(df["C"].unique())
        gamma_vals = sorted(df["gamma"].unique())
        Z = df["score"].values.reshape(len(C_vals), len(gamma_vals))

        cp = plt.contourf(gamma_vals, C_vals, Z, levels=20)
        plt.colorbar(cp)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("gamma")
        plt.ylabel("C")
        plt.title(f"Contour plot ({score_type}) for C and gamma")
        fname = f"contour_grid_{run_id}.png"

    plt.savefig(fname)
    plt.close()
    return fname


def classification_example(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    run_id: str,
    correct: bool = True,
) -> str:
    y_test_flat = y_test.ravel()

    if not correct:
        indices = np.where(y_pred != y_test_flat)[0]
        log.info("Number of misclassified samples: %d", len(indices))
        idx = indices[np.random.randint(0, len(indices))]
        title_str = f"Misclassified {class_labels[y_test_flat[idx]]} as {class_labels[y_pred[idx]]}"
    else:
        indices = np.where(y_pred == y_test_flat)[0]
        log.info("Number of correctly classified samples: %d", len(indices))
        idx = indices[np.random.randint(0, len(indices))]
        title_str = f"Correctly classified {class_labels[y_test_flat[idx]]}"

    plt.imshow(X_test[idx])
    plt.title(title_str)
    fname = f"classification_example_{run_id}.png"
    plt.savefig(fname)
    plt.close()
    return fname


def plot_labels_distribution(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    label_names: list[str],
    save_plot: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, labels, title in zip(axes,
                                  [train_labels, test_labels],
                                  ["Train class distribution (y_train)",
                                   "Test class distribution (y_test)"]):
        sns.countplot(x='label', data=pd.DataFrame(labels, columns=["label"]), ax=ax)
        ax.set_xticklabels(label_names, rotation=45)
        ax.set_title(title)

    plt.tight_layout()
    if save_plot:
        plt.savefig("labels_distro_after_stratified_sampling.png")
    plt.close()


def show_image(dataset: np.ndarray, image_idx: int = 0, grayscale: bool = False) -> None:
    image = dataset[image_idx]
    plt.imshow(image, cmap='gray' if grayscale else None)
    plt.show()


# ── Shared experiment helpers ─────────────────────────────────────────────────

def get_class_labels(dataset_name: str) -> list[str]:
    labels = {
        'CIFAR-10': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck'],
        'MLL': ['ALL', 'AML', 'MLL'],
    }
    return labels[dataset_name]


def grid_search_training(
    grid_parameters: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    k_folds: int = 5,
    Cores: int = -1,
) -> GridSearchCV:
    search = GridSearchCV(
        model, grid_parameters, cv=k_folds,
        scoring='accuracy', n_jobs=Cores, verbose=1,
        return_train_score=True,
    )
    start_time = time.time()
    search.fit(X_train, y_train)
    log.info("Training time: %.2f sec", time.time() - start_time)
    return search


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    return {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, digits=4),
        'y_pred': y_pred,
    }


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_labels: list[str],
    title: str,
) -> str:
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)
    fname = f"conf_matrix_{title}.png"
    plt.savefig(fname)
    plt.close()
    return fname


def move_files_to_output(files: list[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for f in files:
        if os.path.exists(f):
            shutil.move(f, os.path.join(output_dir, f))


def save_and_log_results(
    title: str,
    scaler: str,
    acc_train: float,
    acc_test: float,
    time_taken: float,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    best_params: dict | None = None,
) -> list[str]:
    report = classification_report(y_test, y_pred, digits=4)
    log.info("Test Accuracy: %.4f", acc_test)

    results_data = {
        'Model': title,
        'Scaler': scaler,
        'Train_Accuracy': acc_train,
        'Test_Accuracy': acc_test,
        'Training_Time': time_taken,
        'Report': report,
        'Best_Params': str(best_params) if best_params else "Fixed",
    }

    df = pd.DataFrame([results_data])
    csv_name = f"results_{title}_{scaler}.csv"
    df.to_csv(csv_name, index=False)

    print(tabulate(df[['Model', 'Scaler', 'Test_Accuracy', 'Training_Time']],
                   headers='keys', tablefmt='psql'))

    conf_matrix_fname = plot_confusion_matrix(
        confusion_matrix(y_test, y_pred), class_labels, f"{title}_{scaler}",
    )

    pkl_name = f"model_{title}_{scaler}.pkl"
    joblib.dump(results_data, pkl_name)

    return [csv_name, conf_matrix_fname, pkl_name]
