import json

import numpy as np
import pandas as pd

from scipy.sparse import load_npz, save_npz

from src.config import (
    TRAIN_FEATURES_FILE,
    VALIDATION_FEATURES_FILE,
    TEST_FEATURES_FILE,
    ORIGINAL_VALIDATION_MATRIX_FILE as VALIDATION_MATRIX_FILE,
    ORIGINAL_VALIDATION_LABELS_FILE as VALIDATION_LABELS_FILE,
    ORIGINAL_VALIDATION_CONTEXT_FILE as VALIDATION_CONTEXT_FILE,
    EMBARGO_DIR,
    EMBARGO_VALIDATION_MATRIX_FILE,
    EMBARGO_VALIDATION_LABELS_FILE,
    EMBARGO_VALIDATION_CONTEXT_FILE,
    EMBARGO_EXCLUDED_RUNS_FILE,
    EMBARGO_REPORT_FILE,
)


def load_temporal_data(path):
    columns = [
        "run_id",
        "test_id",
        "timestamp",
        "completion_timestamp",
        "failed",
    ]

    dataframe = pd.read_csv(
        path,
        usecols=columns,
        low_memory=False,
    )

    for column in [
        "timestamp",
        "completion_timestamp",
    ]:
        dataframe[column] = pd.to_datetime(
            dataframe[column],
            utc=True,
            errors="raise",
        )

    if dataframe.empty:
        raise ValueError("Empty dataset.")

    if dataframe.isna().any().any():
        raise ValueError("Missing temporal data.")

    if dataframe.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError("Duplicate test records.")

    if not dataframe["failed"].isin([0, 1]).all():
        raise ValueError("Invalid failure labels.")

    if (
        dataframe["completion_timestamp"]
        < dataframe["timestamp"]
    ).any():
        raise ValueError("Invalid completion timestamp.")

    return dataframe


def find_excluded_runs(dataframe, cutoff):
    completion_times = dataframe.groupby(
        "run_id"
    )["completion_timestamp"].max()

    return completion_times[
        completion_times >= cutoff
    ].index.tolist()


def create_eligibility_mask(dataframe, cutoff):
    excluded_runs = find_excluded_runs(
        dataframe,
        cutoff,
    )

    return ~dataframe["run_id"].isin(
        excluded_runs
    ).to_numpy()


def validate_alignment(features, context, matrix, labels):
    if not (
        len(features)
        == len(context)
        == matrix.shape[0]
        == len(labels)
    ):
        raise ValueError("Dataset row mismatch.")

    for column in ["run_id", "test_id"]:
        if not np.array_equal(
            features[column].to_numpy(),
            context[column].to_numpy(),
        ):
            raise ValueError(
                f"Row alignment mismatch: {column}"
            )

    if not np.array_equal(
        features["failed"].to_numpy(),
        labels,
    ):
        raise ValueError("Feature label mismatch.")

    if not np.array_equal(
        context["failed"].to_numpy(),
        labels,
    ):
        raise ValueError("Context label mismatch.")

    if not np.isfinite(matrix.data).all():
        raise ValueError("Invalid matrix values.")


def main():
    print("=" * 70)
    print("QuantCI - Temporal Embargo Correction")
    print("=" * 70)

    EMBARGO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EMBARGO_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = load_temporal_data(
        TRAIN_FEATURES_FILE
    )

    validation = load_temporal_data(
        VALIDATION_FEATURES_FILE
    )

    test = load_temporal_data(
        TEST_FEATURES_FILE
    )

    validation_start = validation[
        "timestamp"
    ].min()

    test_start = test[
        "timestamp"
    ].min()

    training_excluded = find_excluded_runs(
        train,
        validation_start,
    )

    if training_excluded:
        raise RuntimeError(
            "Training partition requires correction."
        )

    matrix = load_npz(
        VALIDATION_MATRIX_FILE
    ).tocsr()

    labels = np.load(
        VALIDATION_LABELS_FILE
    )

    context = pd.read_csv(
        VALIDATION_CONTEXT_FILE,
        low_memory=False,
    )

    validate_alignment(
        validation,
        context,
        matrix,
        labels,
    )

    mask = create_eligibility_mask(
        validation,
        test_start,
    )

    corrected_matrix = matrix[mask].tocsr()

    corrected_labels = labels[mask]

    corrected_context = context.loc[
        mask
    ].reset_index(drop=True)

    corrected_features = validation.loc[
        mask
    ].reset_index(drop=True)

    excluded_features = validation.loc[
        ~mask
    ].copy()

    if corrected_context.empty:
        raise RuntimeError(
            "No eligible validation records."
        )

    if len(np.unique(corrected_labels)) != 2:
        raise RuntimeError(
            "Corrected validation requires both classes."
        )

    if (
        corrected_features["completion_timestamp"].max()
        >= test_start
    ):
        raise RuntimeError(
            "Temporal boundary violation."
        )

    validate_alignment(
        corrected_features,
        corrected_context,
        corrected_matrix,
        corrected_labels,
    )

    excluded_runs = excluded_features.groupby(
        "run_id"
    ).agg(
        start_time=("timestamp", "min"),
        completion_time=("completion_timestamp", "max"),
        total_tests=("failed", "size"),
        failures=("failed", "sum"),
    ).reset_index()

    save_npz(
        EMBARGO_VALIDATION_MATRIX_FILE,
        corrected_matrix,
    )

    np.save(
        EMBARGO_VALIDATION_LABELS_FILE,
        corrected_labels,
    )

    corrected_context.to_csv(
        EMBARGO_VALIDATION_CONTEXT_FILE,
        index=False,
    )

    excluded_runs.to_csv(
        EMBARGO_EXCLUDED_RUNS_FILE,
        index=False,
    )

    report = {
        "correction_method": "temporal_embargo",
        "validation_start": str(validation_start),
        "test_start": str(test_start),
        "training_rows": len(train),
        "training_boundary_violations": len(
            training_excluded
        ),
        "original_validation_rows": len(validation),
        "original_validation_runs": int(
            validation["run_id"].nunique()
        ),
        "original_validation_failures": int(
            labels.sum()
        ),
        "excluded_validation_rows": int(
            (~mask).sum()
        ),
        "excluded_validation_runs": len(
            excluded_runs
        ),
        "excluded_validation_failures": int(
            labels[~mask].sum()
        ),
        "corrected_validation_rows": len(
            corrected_context
        ),
        "corrected_validation_runs": int(
            corrected_context["run_id"].nunique()
        ),
        "corrected_validation_failures": int(
            corrected_labels.sum()
        ),
        "corrected_validation_failure_rate": float(
            corrected_labels.mean()
        ),
        "corrected_matrix_shape": list(
            corrected_matrix.shape
        ),
        "remaining_validation_boundary_violations": 0,
        "test_rows": len(test),
        "test_failures": int(test["failed"].sum()),
        "original_artifacts_preserved": True,
        "model_reselection_required": True,
        "previous_test_results_observed": True,
    }

    EMBARGO_REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=4))

    print("\nGenerated files:")

    for path in [
        EMBARGO_VALIDATION_MATRIX_FILE,
        EMBARGO_VALIDATION_LABELS_FILE,
        EMBARGO_VALIDATION_CONTEXT_FILE,
        EMBARGO_EXCLUDED_RUNS_FILE,
        EMBARGO_REPORT_FILE,
    ]:
        print(path)

    print("=" * 70)
    print("TEMPORAL EMBARGO CORRECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()