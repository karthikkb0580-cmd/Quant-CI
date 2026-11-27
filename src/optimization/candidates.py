import numpy as np
import pandas as pd

from src.config import (
    FINAL_VALIDATION_PREDICTIONS,
    FINAL_TEST_PREDICTIONS,
    MIN_ESTIMATED_DURATION,
    MAX_CLASSICAL_CANDIDATES,
    OPTIMIZATION_CANDIDATES_FILE,
)


CANDIDATE_COLUMNS = [
    "run_id",
    "test_id",
    "failure_score",
    "estimated_duration_seconds",
]


def validate_candidates(dataframe):
    missing = [
        column
        for column in CANDIDATE_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if dataframe.empty:
        raise ValueError(
            "Candidate dataset is empty."
        )

    if dataframe[
        CANDIDATE_COLUMNS
    ].isna().any().any():
        raise ValueError(
            "Missing candidate values."
        )

    if dataframe.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError(
            "Duplicate candidates."
        )

    scores = dataframe[
        "failure_score"
    ].to_numpy(dtype=float)

    durations = dataframe[
        "estimated_duration_seconds"
    ].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError(
            "Invalid failure scores."
        )

    if not np.isfinite(durations).all():
        raise ValueError(
            "Invalid estimated durations."
        )

    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError(
            "Failure scores must be between 0 and 1."
        )

    if np.any(durations <= 0):
        raise ValueError(
            "Estimated durations must be positive."
        )


def load_predictions(partition="validation"):
    paths = {
        "validation": FINAL_VALIDATION_PREDICTIONS,
        "test": FINAL_TEST_PREDICTIONS,
    }

    if partition not in paths:
        raise ValueError(
            "Partition must be validation or test."
        )

    path = paths[partition]

    if not path.exists():
        raise FileNotFoundError(path)

    dataframe = pd.read_csv(
        path,
        usecols=CANDIDATE_COLUMNS,
        low_memory=False,
    )

    validate_candidates(dataframe)

    return dataframe


def prepare_candidates(
    dataframe,
    run_id,
    max_candidates=None,
):
    validate_candidates(dataframe)

    candidates = dataframe.loc[
        dataframe["run_id"] == run_id,
        CANDIDATE_COLUMNS,
    ].copy()

    if candidates.empty:
        raise ValueError(
            f"Unknown run ID: {run_id}"
        )

    candidates[
        "estimated_duration_seconds"
    ] = candidates[
        "estimated_duration_seconds"
    ].clip(lower=MIN_ESTIMATED_DURATION)

    candidates["risk_per_second"] = (
        candidates["failure_score"]
        / candidates["estimated_duration_seconds"]
    )

    candidates = candidates.sort_values(
        [
            "failure_score",
            "risk_per_second",
            "test_id",
        ],
        ascending=[False, False, True],
    )

    if max_candidates is not None:
        if max_candidates <= 0:
            raise ValueError(
                "Maximum candidates must be positive."
            )

        candidates = candidates.head(
            max_candidates
        )

    return candidates.reset_index(drop=True)


def get_run_ids(dataframe):
    return dataframe[
        "run_id"
    ].drop_duplicates().tolist()


def main():
    dataframe = load_predictions(
        "validation"
    )

    runs = get_run_ids(dataframe)

    partitions = []

    for run_id in runs:
        candidates = prepare_candidates(
            dataframe,
            run_id,
            max_candidates=MAX_CLASSICAL_CANDIDATES,
        )

        partitions.append(candidates)

    combined = pd.concat(
        partitions,
        ignore_index=True,
    )

    OPTIMIZATION_CANDIDATES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OPTIMIZATION_CANDIDATES_FILE,
        index=False,
    )

    print("=" * 60)
    print("QuantCI - Candidate Preparation")
    print("=" * 60)

    print(f"Validation runs: {len(runs)}")
    print(f"Original records: {len(dataframe):,}")
    print(f"Prepared candidates: {len(combined):,}")

    print(
        f"Maximum candidates per run: "
        f"{MAX_CLASSICAL_CANDIDATES}"
    )

    print(
        combined.head(10).to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {OPTIMIZATION_CANDIDATES_FILE}"
    )

    print("CANDIDATE PREPARATION COMPLETE")


if __name__ == "__main__":
    main()