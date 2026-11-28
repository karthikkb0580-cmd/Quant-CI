import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from src.features.multi_repository_history import (
    generate_features,
    FEATURE_COLUMNS,
)

from src.features.multi_repository_matrices import (
    RepositoryPreprocessor,
    COMMIT_NUMERICAL_COLUMNS,
    TEXT_COLUMNS,
    CONTEXT_COLUMNS,
    validate_features,
)


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "multi_repository"

SPLIT_DIR = DATA_DIR / "global_splits"

FEATURE_DIR = DATA_DIR / "global_features"

MATRIX_DIR = DATA_DIR / "global_matrices"

MODEL_DIR = BASE_DIR / "models" / "multi_repository" / "combined"

RESULTS_DIR = BASE_DIR / "results" / "multi_repository" / "combined"

PROJECTS = ["karaf", "hbase"]

PARTITIONS = ["train", "validation", "test"]

KEY_COLUMNS = ["run_id", "test_id"]

HISTORY_COLUMNS = [
    "project",
    "run_id",
    "build_key",
    "stage_id",
    "test_id",
    "timestamp",
    "completion_timestamp",
    "failed",
    "duration_seconds",
]

EXPECTED_COUNTS = {
    "train": (356797, 449),
    "validation": (120251, 88),
    "test": (196729, 108),
}

BATCH_SIZE = 10000


def validate_assignments(assignments):
    required = {
        "project",
        "build_key",
        "partition",
    }

    if not required.issubset(
        assignments.columns
    ):
        raise ValueError(
            "Missing assignment columns"
        )

    if assignments.empty:
        raise ValueError(
            "Empty build assignments"
        )

    if assignments[
        list(required)
    ].isna().any().any():
        raise ValueError(
            "Missing assignment values"
        )

    if assignments.duplicated(
        ["project", "build_key"]
    ).any():
        raise ValueError(
            "Duplicate build assignments"
        )

    if not assignments["partition"].isin(
        PARTITIONS + ["embargoed"]
    ).all():
        raise ValueError(
            "Invalid partition assignment"
        )

    return assignments


def load_assignments():
    path = (
        RESULTS_DIR
        / "global_build_assignments.csv"
    )

    return validate_assignments(
        pd.read_csv(path)
    )


def assign_history_partitions(
    history,
    assignments,
):
    assignments = validate_assignments(
        assignments
    )

    result = history.merge(
        assignments[
            [
                "project",
                "build_key",
                "partition",
            ]
        ],
        on=["project", "build_key"],
        how="left",
        validate="many_to_one",
    )

    if len(result) != len(history):
        raise ValueError(
            "History assignment mismatch"
        )

    if result["partition"].isna().any():
        raise ValueError(
            "Unassigned historical executions"
        )

    return result


def generate_repository_history(
    project,
    assignments,
):
    source = (
        DATA_DIR
        / f"{project}_normalized.csv"
    )

    dataframe = pd.read_csv(
        source,
        usecols=HISTORY_COLUMNS,
    )

    if not dataframe["project"].eq(
        project
    ).all():
        raise ValueError(
            "Repository identity mismatch"
        )

    history = generate_features(
        dataframe
    )

    if len(history) != len(dataframe):
        raise ValueError(
            "Historical row count mismatch"
        )

    history = assign_history_partitions(
        history,
        assignments,
    )

    directory = FEATURE_DIR / project

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for partition in PARTITIONS:
        subset = history[
            history["partition"].eq(
                partition
            )
        ].copy()

        if subset.empty:
            raise ValueError(
                f"Empty {project}/{partition}"
            )

        if subset[FEATURE_COLUMNS].isna().any().any():
            raise ValueError(
                "Missing historical features"
            )

        subset.to_csv(
            directory
            / f"{partition}_history.csv",
            index=False,
        )

        record = {
            "project": project,
            "partition": partition,
            "rows": len(subset),
            "failures": int(
                subset["failed"].sum()
            ),
            "historical_features": len(
                FEATURE_COLUMNS
            ),
            "missing_features": int(
                subset[FEATURE_COLUMNS]
                .isna()
                .sum()
                .sum()
            ),
        }

        records.append(record)

        print(
            json.dumps(record),
            flush=True,
        )

    return records


def merge_partition(
    history,
    commit_data,
):
    if history.duplicated(
        KEY_COLUMNS
    ).any():
        raise ValueError(
            "Duplicate history records"
        )

    if commit_data.duplicated(
        KEY_COLUMNS
    ).any():
        raise ValueError(
            "Duplicate commit records"
        )

    result = history.merge(
        commit_data,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if len(result) != len(commit_data):
        raise ValueError(
            "Partition row count mismatch"
        )

    if not result["_merge"].eq(
        "both"
    ).all():
        raise ValueError(
            "Unmatched partition records"
        )

    result = result.drop(
        columns="_merge"
    )

    return validate_features(
        result
    )


def load_global_partition(
    project,
    partition,
):
    history_path = (
        FEATURE_DIR
        / project
        / f"{partition}_history.csv"
    )

    split_path = (
        SPLIT_DIR
        / project
        / f"{partition}.csv"
    )

    history = pd.read_csv(
        history_path
    )

    commit_data = pd.read_csv(
        split_path,
        usecols=(
            KEY_COLUMNS
            + TEXT_COLUMNS
            + COMMIT_NUMERICAL_COLUMNS
        ),
        keep_default_na=False,
    )

    return merge_partition(
        history,
        commit_data,
    )


def load_combined_partition(
    partition,
):
    frames = []

    for project in PROJECTS:
        print(
            f"Loading {project}/{partition}",
            flush=True,
        )

        frames.append(
            load_global_partition(
                project,
                partition,
            )
        )

    dataframe = pd.concat(
        frames,
        ignore_index=True,
    )

    dataframe["timestamp"] = (
        pd.to_datetime(
            dataframe["timestamp"],
            utc=True,
        )
    )

    dataframe = dataframe.sort_values(
        [
            "timestamp",
            "project",
            "build_key",
            "run_id",
            "test_id",
        ]
    ).reset_index(drop=True)

    return validate_features(
        dataframe
    )


def verify_expected_counts(
    dataframe,
    partition,
):
    expected_rows, expected_failures = (
        EXPECTED_COUNTS[partition]
    )

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"{partition}: expected "
            f"{expected_rows} rows, "
            f"received {len(dataframe)}"
        )

    failures = int(
        dataframe["failed"].sum()
    )

    if failures != expected_failures:
        raise ValueError(
            f"{partition}: expected "
            f"{expected_failures} failures, "
            f"received {failures}"
        )

    return True


def transform_in_batches(
    preprocessor,
    dataframe,
    batch_size=BATCH_SIZE,
):
    if batch_size <= 0:
        raise ValueError(
            "Invalid batch size"
        )

    if dataframe.empty:
        raise ValueError(
            "Empty transformation dataset"
        )

    matrices = []

    for start in range(
        0,
        len(dataframe),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(dataframe),
        )

        batch = dataframe.iloc[
            start:end
        ]

        matrix = preprocessor.transform(
            batch
        )

        matrices.append(matrix)

        print(
            f"Transformed {end:,}"
            f"/{len(dataframe):,}",
            flush=True,
        )

    result = sparse.vstack(
        matrices,
        format="csr",
        dtype=np.float32,
    )

    if not np.isfinite(
        result.data
    ).all():
        raise ValueError(
            "Invalid sparse matrix"
        )

    return result


def save_matrix(
    dataframe,
    matrix,
    partition,
):
    MATRIX_DIR.mkdir(
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
            "Matrix-label count mismatch"
        )

    sparse.save_npz(
        MATRIX_DIR / f"X_{partition}.npz",
        matrix,
        compressed=True,
    )

    np.save(
        MATRIX_DIR / f"y_{partition}.npy",
        labels,
    )

    context.to_csv(
        MATRIX_DIR
        / f"{partition}_context.csv",
        index=False,
    )

    density = matrix.nnz / (
        matrix.shape[0]
        * matrix.shape[1]
    )

    memory_bytes = (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )

    return {
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
        "repository_counts": {
            str(project): int(count)
            for project, count in (
                context["project"]
                .value_counts()
                .items()
            )
        },
    }


def verify_global_contexts(contexts):
    for partition in PARTITIONS:
        if partition not in contexts:
            raise ValueError(
                "Missing partition context"
            )

        if contexts[partition].empty:
            raise ValueError(
                "Empty partition context"
            )

    build_sets = {
        partition: set(
            zip(
                contexts[partition]["project"],
                contexts[partition]["build_key"],
            )
        )
        for partition in PARTITIONS
    }

    if (
        build_sets["train"]
        & build_sets["validation"]
    ):
        raise ValueError(
            "Train-validation build overlap"
        )

    if (
        build_sets["train"]
        & build_sets["test"]
    ):
        raise ValueError(
            "Train-test build overlap"
        )

    if (
        build_sets["validation"]
        & build_sets["test"]
    ):
        raise ValueError(
            "Validation-test build overlap"
        )

    starts = {}

    completions = {}

    for partition, context in contexts.items():
        starts[partition] = pd.to_datetime(
            context["timestamp"],
            utc=True,
        )

        completions[partition] = pd.to_datetime(
            context["completion_timestamp"],
            utc=True,
        )

    if (
        completions["train"].max()
        >= starts["validation"].min()
    ):
        raise ValueError(
            "Global training overlap"
        )

    if (
        completions["validation"].max()
        >= starts["test"].min()
    ):
        raise ValueError(
            "Global validation overlap"
        )

    return True


def main():
    started = time.perf_counter()

    for directory in [
        FEATURE_DIR,
        MATRIX_DIR,
        MODEL_DIR,
        RESULTS_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("=" * 70)

    print("QUANTCI GLOBAL FEATURE GENERATION")

    print("=" * 70)

    assignments = load_assignments()

    history_records = []

    for project in PROJECTS:
        print(
            f"\nGenerating history: {project}"
        )

        history_records.extend(
            generate_repository_history(
                project,
                assignments,
            )
        )

        gc.collect()

    pd.DataFrame(
        history_records
    ).to_csv(
        RESULTS_DIR
        / "global_history_summary.csv",
        index=False,
    )

    contexts = {}

    for partition in PARTITIONS:
        frames = []

        for project in PROJECTS:
            frames.append(
                pd.read_csv(
                    FEATURE_DIR
                    / project
                    / f"{partition}_history.csv",
                    usecols=[
                        "project",
                        "build_key",
                        "timestamp",
                        "completion_timestamp",
                    ],
                )
            )

        contexts[partition] = pd.concat(
            frames,
            ignore_index=True,
        )

    verify_global_contexts(contexts)

    del contexts

    gc.collect()

    preprocessor = RepositoryPreprocessor()

    matrix_records = []

    for partition in PARTITIONS:
        print(
            f"\nProcessing combined {partition}"
        )

        dataframe = load_combined_partition(
            partition
        )

        verify_expected_counts(
            dataframe,
            partition,
        )

        if partition == "train":
            print(
                "Fitting shared preprocessor",
                flush=True,
            )

            preprocessor.fit(
                dataframe
            )

            joblib.dump(
                preprocessor,
                MODEL_DIR
                / "preprocessor.joblib",
            )

            (
                MODEL_DIR
                / "feature_names.json"
            ).write_text(
                json.dumps(
                    preprocessor.feature_names(),
                    indent=4,
                ),
                encoding="utf-8",
            )

        matrix = transform_in_batches(
            preprocessor,
            dataframe,
        )

        record = save_matrix(
            dataframe,
            matrix,
            partition,
        )

        matrix_records.append(record)

        print(
            json.dumps(record, indent=4)
        )

        del dataframe

        del matrix

        gc.collect()

    summary = pd.DataFrame(
        matrix_records
    )

    summary.to_csv(
        RESULTS_DIR
        / "global_matrix_summary.csv",
        index=False,
    )

    report = {
        "projects": PROJECTS,
        "preprocessing": "shared",
        "fit_partition": "global_train",
        "evaluation_mode": "walk_forward",
        "historical_features": len(
            FEATURE_COLUMNS
        ),
        "feature_count": len(
            preprocessor.feature_names()
        ),
        "matrix_format": "csr",
        "matrix_dtype": "float32",
        "global_temporal_isolation": True,
        "total_rows": int(
            summary["rows"].sum()
        ),
        "total_failures": int(
            summary["failures"].sum()
        ),
        "partitions": matrix_records,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    (
        RESULTS_DIR
        / "global_feature_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nGlobal Matrix Summary\n")

    print(
        summary.to_string(index=False)
    )

    print(
        "\nGLOBAL FEATURE GENERATION COMPLETED"
    )


if __name__ == "__main__":
    main()