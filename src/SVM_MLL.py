import logging

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from tabulate import tabulate
import joblib
import os

import Utils

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def run_svm_experiment(X_train, y_train, X_test, y_test,
                       scaler_name, cores, grid_params,
                       pca_threshold, pca_enable, k_folds, class_labels):

    X_train_scaled, X_test_scaled, pca_n_components = Utils.preprocess_mll(
        X_train, X_test,
        scaler_name=scaler_name,
        pca_info_threshold=pca_threshold,
        pca_enable=pca_enable,
    )

    if pca_enable:
        log.info("PCA components to retain %.0f%% variance: %d",
                 pca_threshold * 100, pca_n_components)
    else:
        log.info("PCA disabled")
    log.info("Train shape after preprocessing: %s", X_train_scaled.shape)

    kernel = grid_params['kernel'][0]
    run_id = f"MLL_{kernel}_{scaler_name}_PCA" if pca_enable else f"MLL_{kernel}_{scaler_name}"

    # --- Grid search ---
    search = Utils.grid_search_training(
        grid_params, X_train_scaled, y_train,
        model=SVC(), k_folds=k_folds, Cores=cores,
    )

    results_df = pd.DataFrame(search.cv_results_)
    results_df.sort_values(by='rank_test_score', ascending=True, inplace=True)
    grid_csv = f"results_{run_id}.csv"
    results_df.to_csv(grid_csv, index=False)
    generated_files = [grid_csv]

    n_combinations = len(list(ParameterGrid(search.param_grid)))
    if kernel in ('rbf', 'poly') and n_combinations >= 4:
        generated_files.append(
            Utils.plot_contour_from_grid(run_id, search, score_type="mean_test_score")
        )

    print(tabulate(results_df, headers='keys', tablefmt='psql'))

    # --- Best model evaluation ---
    best_model = search.best_estimator_
    log.info("Best parameters: %s", search.best_params_)

    metrics = Utils.evaluate_model(
        best_model, X_train_scaled, y_train, X_test_scaled, y_test,
    )
    log.info("Train accuracy: %.4f", metrics['train_accuracy'])
    log.info("Test accuracy:  %.4f", metrics['test_accuracy'])

    generated_files.append(
        Utils.plot_confusion_matrix(
            confusion_matrix(y_test, metrics['y_pred']), class_labels, run_id,
        )
    )

    best_est_results = {
        'scaler': scaler_name,
        'best_params': search.best_params_,
        'pca_components': pca_n_components,
        'train_accuracy': metrics['train_accuracy'],
        'test_accuracy': metrics['test_accuracy'],
        'cv_mean_score': search.best_score_,
        'classification_report': metrics['report'],
    }
    best_est_df = pd.DataFrame([best_est_results])
    best_est_csv = f"best_est_results_{run_id}.csv"
    best_est_df.to_csv(best_est_csv, index=False)
    generated_files.append(best_est_csv)

    print(tabulate(best_est_df, headers='keys', tablefmt='psql'))

    model_pkl = f"svm_best_model_{run_id}.pkl"
    joblib.dump(best_model, model_pkl)
    generated_files.append(model_pkl)

    Utils.move_files_to_output(generated_files, f"{run_id}_output_files")


def main():

    cfg = Utils.load_config('MLL')
    PCA_enable = True
    Multiple_cores_enable = False

    class_labels = Utils.get_class_labels('MLL')

    NumOfCores = (
        np.linspace(1, os.cpu_count(), os.cpu_count())
        if Multiple_cores_enable else np.array([-1])
    )

    X_train, y_train, X_test, y_test = Utils.load_mll()
    log.info("Train: %s  Test: %s", X_train.shape, X_test.shape)

    Utils.plot_labels_distribution(y_train, y_test, class_labels, save_plot=False)

    for scaler in cfg['scalers']:
        for cores in NumOfCores:
            run_svm_experiment(
                X_train, y_train, X_test, y_test,
                scaler_name=scaler,
                cores=int(cores),
                grid_params=cfg['svm_grid'],
                pca_threshold=cfg['pca_info_threshold'],
                pca_enable=PCA_enable,
                k_folds=cfg['k_folds_cv'],
                class_labels=class_labels,
            )


if __name__ == '__main__':
    main()
