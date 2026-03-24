import logging
import time

import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

import Utils

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def run_kpca_lda_standalone_grid(
    X_train, y_train, X_test, y_test, k_folds, param_grid,
):
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('kpca', KernelPCA(n_jobs=-1)),  # If crashing, set n_jobs=1.
        ('lda', LinearDiscriminantAnalysis()),
    ])

    search = GridSearchCV(
        pipeline, param_grid, cv=k_folds,
        scoring='accuracy', n_jobs=1, verbose=1,  # n_jobs=1 to be safe with memory
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    training_time = time.time() - start_time
    log.info("Total Grid Search Time: %.2f sec", training_time)

    results_df = pd.DataFrame(search.cv_results_).sort_values(by='rank_test_score')
    print(tabulate(
        results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].head(),
        headers='keys', tablefmt='psql',
    ))

    grid_csv = f"grid_results_KPCA_LDA_{param_grid['kpca__kernel'][0]}.csv"
    results_df.to_csv(grid_csv, index=False)

    best_model = search.best_estimator_
    log.info("Best Parameters: %s", search.best_params_)

    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)

    return best_model, train_acc, test_acc, y_pred, training_time, search.best_params_, grid_csv


def main():

    model_name = "KPCA_LDA_Standalone"
    dataset_name = 'MLL'
    SCALER = "MinMaxScaler"

    cfg = Utils.load_config(dataset_name)
    if dataset_name == 'CIFAR-10':
        X_train, y_train, X_test, y_test = Utils.load_cifar10(sample_size=cfg['sample_size'])
    elif dataset_name == 'MLL':
        X_train, y_train, X_test, y_test = Utils.load_mll()

    class_labels = Utils.get_class_labels(dataset_name)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    log.info("Training Set Shape: %s", X_train.shape)

    grid_parameters = cfg['kpca_standalone_grid']

    best_model, train_acc, test_acc, y_pred, training_time, best_params, grid_csv = \
        run_kpca_lda_standalone_grid(
            X_train, y_train, X_test, y_test, cfg['k_folds_cv'], grid_parameters,
        )

    result_files = Utils.save_and_log_results(
        model_name, SCALER, train_acc, test_acc, training_time,
        y_test, y_pred, class_labels, best_params,
    )

    kernel = grid_parameters['kpca__kernel'][0]
    output_dir = f"{dataset_name}_KPCA_LDA_{kernel}_standalone_output"
    Utils.move_files_to_output([grid_csv] + result_files, output_dir)


if __name__ == '__main__':
    main()
