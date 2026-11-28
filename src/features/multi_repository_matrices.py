import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.features.multi_repository_history import FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "multi_repository"

SPLIT_DIR = DATA_DIR / "splits"

HISTORY_DIR = DATA_DIR / "features"

MATRIX_DIR = DATA_DIR / "matrices"

MODEL_DIR = BASE_DIR / "models" / "multi_repository"

RESULTS_DIR = BASE_DIR / "results" / "multi_repository"

PROJECTS = ["karaf", "hbase"]

PARTITIONS = ["train", "validation", "test"]

KEY_COLUMNS = ["run_id", "test_id"]

TEXT_COLUMNS = [
    "commit_message",
    "changed_files_text",
]

COMMIT_NUMERICAL_COLUMNS = [
    "num_commits",
    "num_changed_files",
    "num_additions",
    "num_deletions",
    "num_changed_lines",
    "num_added_files",
    "num_modified_files",
    "num_removed_files",
]

NUMERICAL_COLUMNS = (
    FEATURE_COLUMNS + COMMIT_NUMERICAL_COLUMNS
)

CONTEXT_COLUMNS = [
    "project",
    "run_id",
    "build_key",
    "stage_id",
    "timestamp",
    "test_id",
    "failed",
    "duration_seconds",
    "estimated_duration_seconds",
    "historical_failure_rate",
    "smoothed_failure_rate",
]


def tokenize_identifier(value):
    import re

    value = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        str(value),
    )

    return re.findall(
        r"[a-z0-9]+",
        value.lower(),
    )


def normalize_text(value):
    if pd.isna(value):
        return ""

    return " ".join(
        tokenize_identifier(value)
    )


def validate_features(dataframe):
    required = set(
        KEY_COLUMNS
        + TEXT_COLUMNS
        + NUMERICAL_COLUMNS
        + CONTEXT_COLUMNS
    )

    missing = required.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if dataframe.empty:
        raise ValueError("Empty dataset")

    if dataframe.duplicated(
        KEY_COLUMNS
    ).any():
        raise ValueError(
            "Duplicate test executions"
        )

    if not dataframe["failed"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "Invalid failure labels"
        )

    values = dataframe[
        NUMERICAL_COLUMNS
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Invalid numerical features"
        )

    return dataframe


def load_partition(project, partition):
    split_path = (
        SPLIT_DIR
        / project
        / f"{partition}.csv"
    )

    history_path = (
        HISTORY_DIR
        / project
        / f"{partition}_history.csv"
    )

    split = pd.read_csv(
        split_path,
        usecols=(
            KEY_COLUMNS
            + TEXT_COLUMNS
            + COMMIT_NUMERICAL_COLUMNS
        ),
        keep_default_na=False,
    )

    history = pd.read_csv(
        history_path
    )

    dataframe = history.merge(
        split,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if len(dataframe) != len(split):
        raise ValueError(
            "Partition row count mismatch"
        )

    if not dataframe["_merge"].eq(
        "both"
    ).all():
        raise ValueError(
            "Unmatched historical records"
        )

    dataframe = dataframe.drop(
        columns="_merge"
    )

    return validate_features(
        dataframe
    )


def prepare_text(dataframe):
    messages = dataframe[
        "commit_message"
    ].map(normalize_text)

    files = dataframe[
        "changed_files_text"
    ].map(normalize_text)

    tests = dataframe[
        "test_id"
    ].map(normalize_text)

    combined = messages + " " + files

    return combined, messages, files, tests


def token_similarity(left, right):
    left_tokens = set(left.split())

    right_tokens = set(right.split())

    union = left_tokens | right_tokens

    if not union:
        return 0.0

    return len(
        left_tokens & right_tokens
    ) / len(union)


def similarity_features(
    messages,
    files,
    tests,
):
    values = np.empty(
        (len(tests), 2),
        dtype=np.float32,
    )

    for index, (message, file_text, test) in enumerate(
        zip(messages, files, tests)
    ):
        values[index, 0] = token_similarity(
            message,
            test,
        )

        values[index, 1] = token_similarity(
            file_text,
            test,
        )

    return sparse.csr_matrix(values)


class RepositoryPreprocessor:
    def __init__(self):
        self.code_vectorizer = TfidfVectorizer(
            max_features=500,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
            sublinear_tf=True,
        )

        self.test_vectorizer = TfidfVectorizer(
            max_features=200,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
            sublinear_tf=True,
        )

        self.scaler = StandardScaler()

        self.fitted = False

    def fit(self, dataframe):
        validate_features(dataframe)

        combined, _, _, tests = prepare_text(
            dataframe
        )

        self.code_vectorizer.fit(combined)

        self.test_vectorizer.fit(tests)

        self.scaler.fit(
            dataframe[NUMERICAL_COLUMNS]
        )

        self.fitted = True

        return self

    def transform(self, dataframe):
        if not self.fitted:
            raise ValueError(
                "Preprocessor has not been fitted"
            )

        validate_features(dataframe)

        combined, messages, files, tests = (
            prepare_text(dataframe)
        )

        code_matrix = (
            self.code_vectorizer.transform(
                combined
            )
        )

        test_matrix = (
            self.test_vectorizer.transform(
                tests
            )
        )

        numerical = self.scaler.transform(
            dataframe[NUMERICAL_COLUMNS]
        ).astype(np.float32)

        numerical_matrix = sparse.csr_matrix(
            numerical
        )

        similarities = similarity_features(
            messages,
            files,
            tests,
        )

        matrix = sparse.hstack(
            [
                code_matrix,
                test_matrix,
                numerical_matrix,
                similarities,
            ],
            format="csr",
            dtype=np.float32,
        )

        if not np.isfinite(
            matrix.data
        ).all():
            raise ValueError(
                "Non-finite matrix values"
            )

        return matrix

    def fit_transform(self, dataframe):
        return self.fit(
            dataframe
        ).transform(dataframe)

    def feature_names(self):
        if not self.fitted:
            raise ValueError(
                "Preprocessor has not been fitted"
            )

        code_names = [
            f"code_tfidf_{name}"
            for name in self.code_vectorizer.get_feature_names_out()
        ]

        test_names = [
            f"test_tfidf_{name}"
            for name in self.test_vectorizer.get_feature_names_out()
        ]

        return (
            code_names
            + test_names
            + NUMERICAL_COLUMNS
            + [
                "commit_test_similarity",
                "files_test_similarity",
            ]
        )


def save_partition(
    dataframe,
    matrix,
    project,
    partition,
):
    directory = MATRIX_DIR / project

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = dataframe[
        "failed"
    ].to_numpy(dtype=np.int8)

    context = dataframe[
        CONTEXT_COLUMNS
    ].copy()

    if matrix.shape[0] != len(labels):
        raise ValueError(
            "Matrix and label count mismatch"
        )

    sparse.save_npz(
        directory / f"X_{partition}.npz",
        matrix,
        compressed=True,
    )

    np.save(
        directory / f"y_{partition}.npy",
        labels,
    )

    context.to_csv(
        directory / f"{partition}_context.csv",
        index=False,
    )

    density = matrix.nnz / (
        matrix.shape[0] * matrix.shape[1]
    )

    memory_bytes = (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )

    return {
        "project": project,
        "partition": partition,
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "failures": int(labels.sum()),
        "nonzero_values": int(matrix.nnz),
        "density": float(density),
        "sparsity": float(1 - density),
        "memory_mb": float(
            memory_bytes / 1024 ** 2
        ),
        "dtype": str(matrix.dtype),
    }


def process_project(project):
    started = time.perf_counter()

    model_directory = MODEL_DIR / project

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    preprocessor = RepositoryPreprocessor()

    reports = []

    for partition in PARTITIONS:
        print(
            f"\nLoading {project}/{partition}"
        )

        dataframe = load_partition(
            project,
            partition,
        )

        print(
            f"Executions: {len(dataframe):,}"
        )

        if partition == "train":
            matrix = preprocessor.fit_transform(
                dataframe
            )
        else:
            matrix = preprocessor.transform(
                dataframe
            )

        report = save_partition(
            dataframe,
            matrix,
            project,
            partition,
        )

        reports.append(report)

        print(
            json.dumps(report, indent=4)
        )

        del dataframe
        del matrix

        gc.collect()

    feature_names = (
        preprocessor.feature_names()
    )

    joblib.dump(
        preprocessor,
        model_directory / "preprocessor.joblib",
    )

    (
        model_directory / "feature_names.json"
    ).write_text(
        json.dumps(
            feature_names,
            indent=4,
        ),
        encoding="utf-8",
    )

    return {
        "project": project,
        "feature_count": len(feature_names),
        "fit_partition": "train",
        "code_tfidf_features": len(
            preprocessor.code_vectorizer.vocabulary_
        ),
        "test_tfidf_features": len(
            preprocessor.test_vectorizer.vocabulary_
        ),
        "numerical_features": len(
            NUMERICAL_COLUMNS
        ),
        "similarity_features": 2,
        "partitions": reports,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = []

    print("=" * 70)
    print("QUANTCI MULTI-REPOSITORY MATRICES")
    print("=" * 70)

    for project in PROJECTS:
        print(f"\nProcessing {project}")

        reports.append(
            process_project(project)
        )

        gc.collect()

    summary = pd.DataFrame([
        partition
        for report in reports
        for partition in report["partitions"]
    ])

    summary.to_csv(
        RESULTS_DIR / "matrix_summary.csv",
        index=False,
    )

    report = {
        "projects": PROJECTS,
        "preprocessing_fit_partition": "train",
        "matrix_format": "csr",
        "matrix_dtype": "float32",
        "repositories": reports,
    }

    (
        RESULTS_DIR / "matrix_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nFeature Matrix Summary\n")

    print(
        summary.to_string(index=False)
    )

    print(
        "\nMULTI-REPOSITORY MATRIX GENERATION COMPLETED"
    )


if __name__ == "__main__":
    main()