import re
import json

from collections import Counter
from zipfile import ZipFile

import pandas as pd

from src.config import (
    TEST_RESULTS_DIR,
    COMMIT_DATA_DIR,
    METADATA_FILE,
    PROJECT_NAME,
    RESULTS_DIR,
)


def parse_archive_key(path):
    relative = path.relative_to(TEST_RESULTS_DIR)

    if len(relative.parts) != 3:
        raise ValueError(f"Unexpected archive path: {relative}")

    build_folder, stage_folder, _ = relative.parts

    match = re.fullmatch(
        r"(PR-\d+)_build(\d+)",
        build_folder,
    )

    if not match:
        raise ValueError(f"Invalid build folder: {build_folder}")

    if not stage_folder.startswith("stage_"):
        raise ValueError(f"Invalid stage folder: {stage_folder}")

    return (
        match.group(1),
        int(match.group(2)),
        stage_folder.removeprefix("stage_"),
    )


def load_test_archive(path):
    with ZipFile(path) as archive:
        with archive.open("test_class.csv") as source:
            return pd.read_csv(source)


def main():
    print("=" * 60)
    print("QuantCI - Dataset Validation")
    print("=" * 60)

    metadata = pd.read_csv(
        METADATA_FILE,
        dtype={"stage_id": str},
    )

    metadata = metadata[
        metadata["project"] == PROJECT_NAME
    ].copy()

    metadata["build_id"] = metadata["build_id"].astype(int)

    key_columns = [
        "pr_name",
        "build_id",
        "stage_id",
    ]

    if metadata.duplicated(key_columns).any():
        raise ValueError("Duplicate metadata keys found.")

    metadata_index = metadata.set_index(key_columns)

    metadata_keys = set(metadata_index.index)

    archives = sorted(
        TEST_RESULTS_DIR.rglob("test_class.csv.zip")
    )

    archive_keys = set()

    outcome_counts = Counter()
    matched_outcome_counts = Counter()
    previous_outcome_counts = Counter()

    total_rows = 0
    matched_rows = 0
    missing_test_names = 0
    invalid_durations = 0
    missing_outcomes = 0
    duplicate_rows = 0
    missing_commit_data = 0

    duration_ratios = []
    label_comparisons = []
    unmatched_archives = []

    for path in archives:
        key = parse_archive_key(path)

        if key in archive_keys:
            raise ValueError(f"Duplicate archive key: {key}")

        archive_keys.add(key)

        dataframe = load_test_archive(path)

        required_columns = {
            "testclass",
            "duration",
            "outcome",
            "last_outcome",
        }

        if not required_columns.issubset(dataframe.columns):
            raise ValueError(f"Missing columns in {path}")

        total_rows += len(dataframe)

        counts = dataframe["outcome"].value_counts()

        outcome_counts.update(counts.to_dict())

        previous_outcome_counts.update(
            dataframe["last_outcome"].value_counts().to_dict()
        )

        missing_test_names += int(
            dataframe["testclass"].isna().sum()
        )

        missing_outcomes += int(
            dataframe["outcome"].isna().sum()
        )

        duration = pd.to_numeric(
            dataframe["duration"],
            errors="coerce",
        )

        invalid_durations += int(
            (
                duration.isna()
                | (duration < 0)
                | duration.isin([float("inf"), float("-inf")])
            ).sum()
        )

        duplicate_rows += int(
            dataframe.duplicated("testclass").sum()
        )

        build_folder = f"{key[0]}_build{key[1]}"

        commit_folder = COMMIT_DATA_DIR / build_folder

        commit_files = list(
            commit_folder.glob("*.json")
        )

        if not commit_files:
            missing_commit_data += 1

        if key not in metadata_keys:
            unmatched_archives.append(
                str(path.relative_to(TEST_RESULTS_DIR))
            )
            continue

        matched_rows += len(dataframe)

        matched_outcome_counts.update(counts.to_dict())

        row = metadata_index.loc[key]

        label_comparisons.append(
            {
                "pr_name": key[0],
                "build_id": key[1],
                "stage_id": key[2],
                "outcome_counts": {
                    str(label): int(count)
                    for label, count in counts.items()
                },
                "metadata_passed": int(row["num_pass_class"]),
                "metadata_failed": int(row["num_fail_class"]),
            }
        )

        reference_duration = row["test_suite_duration_s"]

        if pd.notna(reference_duration) and reference_duration > 0:
            duration_ratios.append(
                float(duration.sum() / reference_duration)
            )

    matched_keys = archive_keys & metadata_keys

    report = {
        "metadata_runs": len(metadata_keys),
        "test_archives": len(archives),
        "matched_runs": len(matched_keys),
        "unmatched_archives": len(archive_keys - metadata_keys),
        "metadata_without_archive": len(metadata_keys - archive_keys),
        "total_test_rows": total_rows,
        "matched_test_rows": matched_rows,
        "outcome_counts": {
            str(key): int(value)
            for key, value in outcome_counts.items()
        },
        "matched_outcome_counts": {
            str(key): int(value)
            for key, value in matched_outcome_counts.items()
        },
        "previous_outcome_counts": {
            str(key): int(value)
            for key, value in previous_outcome_counts.items()
        },
        "missing_test_names": missing_test_names,
        "missing_outcomes": missing_outcomes,
        "invalid_durations": invalid_durations,
        "duplicate_test_classes": duplicate_rows,
        "missing_commit_data": missing_commit_data,
        "median_duration_ratio": (
            float(pd.Series(duration_ratios).median())
            if duration_ratios
            else None
        ),
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = RESULTS_DIR / "dataset_validation.json"

    report_file.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    comparison_file = RESULTS_DIR / "label_comparison.json"

    comparison_file.write_text(
        json.dumps(label_comparisons, indent=4),
        encoding="utf-8",
    )

    unmatched_file = RESULTS_DIR / "unmatched_archives.json"

    unmatched_file.write_text(
        json.dumps(unmatched_archives, indent=4),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=4))

    print("\nSample label comparisons:")

    for comparison in label_comparisons[:5]:
        print(comparison)

    print(f"\nValidation report: {report_file}")
    print(f"Label comparison: {comparison_file}")
    print(f"Unmatched archives: {unmatched_file}")

    print("\n" + "=" * 60)
    print("DATASET VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()