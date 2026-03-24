import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import Utils

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def run_kpca_lda_plus_classifier(
    X_train, y_train, X_test, y_test, dataset_name,
    classifier_type='knn', kpca_components=50,
    kernel_gamma=None, svm_kernel=None,
):
    log.info("Running KPCA + LDA + %s with MinMaxScaler", classifier_type.upper())

    if dataset_name == 'CIFAR-10':
        knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
        kpca_set = ('kpca', KernelPCA(n_components=kpca_components,
                                       kernel='rbf', gamma=kernel_gamma, n_jobs=-1))
    elif dataset_name == 'MLL':
        knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        kpca_set = ('kpca', KernelPCA(n_components=kpca_components,
                                       kernel='linear', n_jobs=-1))

    if classifier_type == 'knn':
        clf = knn
    elif classifier_type == 'svm':
        clf = SVC(kernel=svm_kernel)
    else:
        clf = KNeighborsClassifier()

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        kpca_set,
        ('lda', LinearDiscriminantAnalysis()),
        ('clf', clf),
    ])

    log.info("Fitting Pipeline (kpca n_components=%d, svm_kernel=%s)...",
             kpca_components, svm_kernel)
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    log.info("Training time: %.2f sec", training_time)

    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)

    return pipeline, train_acc, test_acc, y_pred, training_time


def visualize_LDA_space(X_test, y_test, pipeline, class_labels, dataset_name, classifier_type):
    if dataset_name != 'MLL':
        return None

    X_lda = pipeline[:-1].transform(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    colors = sns.color_palette("hls", len(class_labels))

    for i, label in enumerate(class_labels):
        ax1.scatter(X_lda[y_test == i, 0], X_lda[y_test == i, 1],
                    color=colors[i], label=label, s=50, edgecolors='k', alpha=0.7)
    ax1.set_title(f'Data in LDA Space ({dataset_name})')
    ax1.set_xlabel('LDA Component 1')
    ax1.set_ylabel('LDA Component 2')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    h = .05  # mesh step size
    x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
    y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pipeline.named_steps['clf'].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax2.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    for i, label in enumerate(class_labels):
        ax2.scatter(X_lda[y_test == i, 0], X_lda[y_test == i, 1],
                    color=colors[i], label=label, s=50, edgecolors='k')
    ax2.set_title(f'Decision Boundaries ({classifier_type.upper()})')
    ax2.set_xlabel('LDA Component 1')
    ax2.set_ylabel('LDA Component 2')

    img_name = f"visualization_{dataset_name}_{classifier_type}.png"
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()
    return img_name


def main():

    model_name = "KPCA_LDA_SVM"
    # Options: "KPCA_LDA_KNN", "KPCA_LDA_SVM"

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

    if model_name == "KPCA_LDA_KNN":
        FINAL_CLASSIFIER_NAME = 'knn'
        pipe, train_acc, test_acc, y_pred, training_time = run_kpca_lda_plus_classifier(
            X_train, y_train, X_test, y_test, dataset_name,
            classifier_type=FINAL_CLASSIFIER_NAME,
            kpca_components=cfg['kpca_best_n_components'],
            kernel_gamma=cfg['kpca_best_gamma'],
        )

    elif model_name == "KPCA_LDA_SVM":
        FINAL_CLASSIFIER_NAME = 'svm'
        SVM_KERNEL = 'linear'
        pipe, train_acc, test_acc, y_pred, training_time = run_kpca_lda_plus_classifier(
            X_train, y_train, X_test, y_test, dataset_name,
            classifier_type=FINAL_CLASSIFIER_NAME,
            kpca_components=cfg['kpca_best_n_components'],
            kernel_gamma=cfg['kpca_best_gamma'],
            svm_kernel=SVM_KERNEL,
        )

    result_files = Utils.save_and_log_results(
        model_name, SCALER, train_acc, test_acc, training_time,
        y_test, y_pred, class_labels,
    )

    viz_file = visualize_LDA_space(
        X_test, y_test, pipe, class_labels, dataset_name, FINAL_CLASSIFIER_NAME,
    )
    if viz_file:
        result_files.append(viz_file)

    output_dir = f"{dataset_name}_KPCA_LDA_{FINAL_CLASSIFIER_NAME.upper()}_output"
    Utils.move_files_to_output(result_files, output_dir)


if __name__ == '__main__':
    main()
