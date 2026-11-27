import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import save_npz

from src.config import (
    MATRICES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    TRAIN_FEATURES_FILE,
    VALIDATION_FEATURES_FILE,
    TEST_FEATURES_FILE,
    TRAIN_MATRIX_FILE,
    VALIDATION_MATRIX_FILE,
    TEST_MATRIX_FILE,
    TRAIN_LABELS_FILE,
    VALIDATION_LABELS_FILE,
    TEST_LABELS_FILE,
    TRAIN_CONTEXT_FILE,
    VALIDATION_CONTEXT_FILE,
    TEST_CONTEXT_FILE,
    PREPROCESSOR_FILE,
    FEATURE_NAMES_FILE,
    TFIDF_REPORT_FILE,
)

from src.features.text_features import (
    FeaturePreprocessor,
    NUMERICAL_COLUMNS,
    TEXT_COLUMNS,
    SIMILARITY_COLUMNS,
)


CONTEXT_COLUMNS = [
    "project",
    "run_id",
    "build_key",
    "timestamp",
    "commit_sha",
    "test_id",
    "failed",
    "duration_seconds",
    "estimated_duration_seconds",
    "historical_failure_rate",
    "smoothed_failure_rate",
]


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(path)

    columns = list(
        dict.fromkeys(
            NUMERICAL_COLUMNS
            + TEXT_COLUMNS
            + CONTEXT_COLUMNS
        )
    )

    dataframe = pd.read_csv(
        path,
        usecols=columns,
        keep_default_na=False,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Empty dataset: {path}"
        )

    if not dataframe["failed"].isin([0, 1]).all():
        raise ValueError(
            "Invalid target labels."
        )

    return dataframe


def validate_matrix(matrix, labels, feature_names):
    if matrix.shape[0] != len(labels):
        raise ValueError(
            "Matrix and label row mismatch."
        )

    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            "Feature count mismatch."
        )

    if not np.isfinite(matrix.data).all():
        raise ValueError(
            "Non-finite matrix values."
        )

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(
            "Invalid target labels."
        )

    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Both target classes are required."
        )


def matrix_statistics(matrix, labels):
    rows, columns = matrix.shape

    total_elements = rows * columns

    sparse_bytes = (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )

    dense_bytes = (
        total_elements
        * np.dtype(np.float32).itemsize
    )

    return {
        "rows": int(rows),
        "features": int(columns),
        "nonzero_elements": int(matrix.nnz),
        "density_percent": round(
            matrix.nnz / total_elements * 100,
            4,
        ),
        "sparsity_percent": round(
            (1 - matrix.nnz / total_elements) * 100,
            4,
        ),
        "sparse_memory_mb": round(
            sparse_bytes / (1024 ** 2),
            2,
        ),
        "dense_memory_mb": round(
            dense_bytes / (1024 ** 2),
            2,
        ),
        "passed": int(
            np.sum(labels == 0)
        ),
        "failed": int(
            np.sum(labels == 1)
        ),
    }


def process_partition(
    name,
    dataframe,
    preprocessor,
    matrix_path,
    labels_path,
    context_path,
):
    print(f"\nProcessing {name}")

    matrix = preprocessor.transform(
        dataframe
    )

    labels = dataframe[
        "failed"
    ].to_numpy(
        dtype=np.int8
    )

    feature_names = (
        preprocessor.get_feature_names()
    )

    validate_matrix(
        matrix,
        labels,
        feature_names,
    )

    save_npz(
        matrix_path,
        matrix,
        compressed=True,
    )

    np.save(
        labels_path,
        labels,
        allow_pickle=False,
    )

    dataframe[
        CONTEXT_COLUMNS
    ].to_csv(
        context_path,
        index=False,
    )

    statistics = matrix_statistics(
        matrix,
        labels,
    )

    print(
        f"Matrix shape: {matrix.shape}"
    )

    print(
        f"Failed tests: {statistics['failed']}"
    )

    print(
        f"Sparse memory: "
        f"{statistics['sparse_memory_mb']} MB"
    )

    print(
        f"Saved: {matrix_path}"
    )

    return statistics


def main():
    print("=" * 60)
    print("QuantCI - TF-IDF Feature Engineering")
    print("=" * 60)

    MATRICES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = load_dataset(
        TRAIN_FEATURES_FILE
    )

    print(
        f"\nTraining records: {len(train):,}"
    )

    print(
        "Fitting preprocessor using training data only..."
    )

    preprocessor = FeaturePreprocessor()

    preprocessor.fit(train)

    feature_names = (
        preprocessor.get_feature_names()
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE,
        compress=3,
    )

    FEATURE_NAMES_FILE.write_text(
        json.dumps(
            feature_names,
            indent=4,
        ),
        encoding="utf-8",
    )

    report = {
        "numerical_features": len(
            NUMERICAL_COLUMNS
        ),
        "code_tfidf_features": len(
            preprocessor.code_vectorizer.vocabulary_
        ),
        "test_tfidf_features": len(
            preprocessor.test_vectorizer.vocabulary_
        ),
        "similarity_features": len(
            SIMILARITY_COLUMNS
        ),
        "total_features": len(
            feature_names
        ),
        "fit_partition": "training",
        "matrix_format": "CSR",
        "matrix_dtype": "float32",
    }

    report["training"] = process_partition(
        "Training",
        train,
        preprocessor,
        TRAIN_MATRIX_FILE,
        TRAIN_LABELS_FILE,
        TRAIN_CONTEXT_FILE,
    )

    del train

    validation = load_dataset(
        VALIDATION_FEATURES_FILE
    )

    report["validation"] = process_partition(
        "Validation",
        validation,
        preprocessor,
        VALIDATION_MATRIX_FILE,
        VALIDATION_LABELS_FILE,
        VALIDATION_CONTEXT_FILE,
    )

    del validation

    test = load_dataset(
        TEST_FEATURES_FILE
    )

    report["testing"] = process_partition(
        "Testing",
        test,
        preprocessor,
        TEST_MATRIX_FILE,
        TEST_LABELS_FILE,
        TEST_CONTEXT_FILE,
    )

    del test

    report["total_rows"] = sum(
        report[name]["rows"]
        for name in [
            "training",
            "validation",
            "testing",
        ]
    )

    report["total_failures"] = sum(
        report[name]["failed"]
        for name in [
            "training",
            "validation",
            "testing",
        ]
    )

    TFIDF_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\nFeature Matrix Summary:")

    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    print(
        f"\nPreprocessor: {PREPROCESSOR_FILE}"
    )

    print(
        f"Feature names: {FEATURE_NAMES_FILE}"
    )

    print(
        f"Report: {TFIDF_REPORT_FILE}"
    )

    print("\n" + "=" * 60)
    print("TF-IDF FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()