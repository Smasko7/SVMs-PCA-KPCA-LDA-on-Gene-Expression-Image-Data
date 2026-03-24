import logging
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from tabulate import tabulate
import joblib

import Utils

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def KNN_search(dataset_name, X_train, y_train, X_test, y_test,
               pca_info_threshold, k_folds, scalers, pca_enable, class_labels):

    log.info("Train: %s  Test: %s", X_train.shape, X_test.shape)

    for scaler in scalers:

        if dataset_name == 'CIFAR-10':
            X_train_scaled, X_test_scaled, pca_n_components = Utils.preprocess_cifar10(
                X_train, X_test, scaler_name=scaler,
                pca_info_threshold=pca_info_threshold, pca_enable=pca_enable,
            )
        elif dataset_name == 'MLL':
            X_train_scaled, X_test_scaled, pca_n_components = Utils.preprocess_mll(
                X_train, X_test, scaler_name=scaler,
                pca_info_threshold=pca_info_threshold, pca_enable=pca_enable,
            )

        if pca_enable:
            log.info("PCA components to retain %.0f%% variance: %d",
                     pca_info_threshold * 100, pca_n_components)
        else:
            log.info("PCA disabled")
        log.info("Train shape after preprocessing: %s", X_train_scaled.shape)

        grid_parameters = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}
        run_id = f"{dataset_name}_KNN_{scaler}_PCA" if pca_enable else f"{dataset_name}_KNN_{scaler}"

        search = Utils.grid_search_training(
            grid_parameters, X_train_scaled, y_train,
            model=KNeighborsClassifier(), k_folds=k_folds,
        )

        results_df = pd.DataFrame(search.cv_results_)
        results_df.sort_values(by='rank_test_score', ascending=True, inplace=True)
        grid_csv = f"results_{run_id}.csv"
        results_df.to_csv(grid_csv, index=False)
        generated_files = [grid_csv]

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
            'scaler': scaler,
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

        if dataset_name == 'CIFAR-10':
            generated_files.append(
                Utils.classification_example(
                    X_test, y_test, metrics['y_pred'], class_labels, run_id, correct=False,
                )
            )

        model_pkl = f"knn_best_model_{run_id}.pkl"
        joblib.dump(best_model, model_pkl)
        generated_files.append(model_pkl)

        Utils.move_files_to_output(generated_files, f"{run_id}_output_files")


def NCC_search(dataset_name, X_train, y_train, X_test, y_test,
               pca_info_threshold, scalers, pca_enable, class_labels):

    log.info("Train: %s  Test: %s", X_train.shape, X_test.shape)

    results = []
    best_test_accuracy = -1
    best_model_result = None
    best_y_pred = None

    for scaler in scalers:

        if dataset_name == 'CIFAR-10':
            X_train_scaled, X_test_scaled, pca_n_components = Utils.preprocess_cifar10(
                X_train, X_test, scaler_name=scaler,
                pca_info_threshold=pca_info_threshold, pca_enable=pca_enable,
            )
        elif dataset_name == 'MLL':
            X_train_scaled, X_test_scaled, pca_n_components = Utils.preprocess_mll(
                X_train, X_test, scaler_name=scaler,
                pca_info_threshold=pca_info_threshold, pca_enable=pca_enable,
            )

        if pca_enable:
            log.info("PCA components to retain %.0f%% variance: %d",
                     pca_info_threshold * 100, pca_n_components)
            run_id = f"{dataset_name}_NCC_PCA"
        else:
            log.info("PCA disabled")
            run_id = f"{dataset_name}_NCC"

        clf = NearestCentroid()
        start_time = time.time()
        clf.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        pred_start = time.time()
        y_pred = clf.predict(X_test_scaled)
        pred_time = time.time() - pred_start

        test_acc = accuracy_score(y_test, y_pred)
        log.info("NCC (%s) test accuracy: %.4f", scaler, test_acc)

        res = {
            'Model': f"NCC ({scaler})",
            'Test Accuracy': test_acc,
            'PCA Components': pca_n_components,
            'Training Time (s)': train_time,
            'Prediction Time (s)': pred_time,
        }
        results.append(res)

        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            best_model_result = res
            best_y_pred = y_pred

    results_df = pd.DataFrame(results).sort_values(by='Test Accuracy', ascending=False)
    log.info("NCC Scaler Comparison Results:\n%s",
             tabulate(results_df, headers='keys', tablefmt='psql', floatfmt=".4f"))

    # Confusion matrix for the best NCC model
    name = best_model_result['Model']
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    conf_matrix_fname = f"{dataset_name}_conf_matrix_Best_NCC.png"
    plt.savefig(conf_matrix_fname)
    plt.close()
    log.info("Saved confusion matrix for best model: %s", name)

    Utils.move_files_to_output([conf_matrix_fname], f"{run_id}_output_files")

    return best_model_result


def main():

    dataset_name = 'CIFAR-10'
    PCA_enable = True

    cfg = Utils.load_config(dataset_name)
    if dataset_name == 'CIFAR-10':
        X_train, y_train, X_test, y_test = Utils.load_cifar10(sample_size=cfg['sample_size'])
    elif dataset_name == 'MLL':
        X_train, y_train, X_test, y_test = Utils.load_mll()

    class_labels = Utils.get_class_labels(dataset_name)

    KNN_search(dataset_name, X_train, y_train, X_test, y_test,
               cfg['pca_info_threshold'], cfg['k_folds_cv'],
               cfg['scalers'], PCA_enable, class_labels)

    NCC_search(dataset_name, X_train, y_train, X_test, y_test,
               cfg['pca_info_threshold'], cfg['scalers'], PCA_enable, class_labels)


if __name__ == "__main__":
    main()
