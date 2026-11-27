import json
import re

from zipfile import ZipFile

import pandas as pd

from src.config import (
    PROJECT_NAME,
    METADATA_FILE,
    TEST_RESULTS_DIR,
    COMMIT_DATA_DIR,
    NORMALIZED_DATA_FILE,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)


def load_metadata():
    dataframe = pd.read_csv(
        METADATA_FILE,
        dtype={"stage_id": str},
    )

    dataframe = dataframe[
        dataframe["project"] == PROJECT_NAME
    ].copy()

    dataframe["build_id"] = dataframe["build_id"].astype(int)

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["build_timestamp"],
        unit="s",
        utc=True,
        errors="raise",
    )

    key_columns = [
        "pr_name",
        "build_id",
        "stage_id",
    ]

    if dataframe.duplicated(key_columns).any():
        raise ValueError("Duplicate metadata keys found.")

    return dataframe.sort_values(
        ["timestamp", "pr_name", "build_id", "stage_id"]
    ).reset_index(drop=True)


def load_test_results(pr_name, build_id, stage_id):
    build_folder = f"{pr_name}_build{build_id}"

    archive_path = (
        TEST_RESULTS_DIR
        / build_folder
        / f"stage_{stage_id}"
        / "test_class.csv.zip"
    )

    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    with ZipFile(archive_path) as archive:
        with archive.open("test_class.csv") as source:
            dataframe = pd.read_csv(source)

    required_columns = {
        "testclass",
        "duration",
        "outcome",
        "last_outcome",
    }

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            f"Missing required columns: {archive_path}"
        )

    dataframe = dataframe.rename(
        columns={
            "testclass": "test_id",
            "duration": "duration_seconds",
            "outcome": "failed",
            "last_outcome": "previous_outcome_raw",
        }
    )

    dataframe["test_id"] = (
        dataframe["test_id"].astype("string").str.strip()
    )

    dataframe["duration_seconds"] = pd.to_numeric(
        dataframe["duration_seconds"],
        errors="raise",
    )

    dataframe["failed"] = pd.to_numeric(
        dataframe["failed"],
        errors="raise",
    )

    if dataframe["test_id"].isna().any():
        raise ValueError("Missing test identifiers.")

    if dataframe["test_id"].eq("").any():
        raise ValueError("Empty test identifiers.")

    if not dataframe["failed"].isin([0, 1]).all():
        raise ValueError("Unexpected outcome labels.")

    if (
        dataframe["duration_seconds"].isna().any()
        or dataframe["duration_seconds"].lt(0).any()
        or dataframe["duration_seconds"].isin(
            [float("inf"), float("-inf")]
        ).any()
    ):
        raise ValueError("Invalid execution durations.")

    if dataframe.duplicated("test_id").any():
        raise ValueError("Duplicate test identifiers.")

    dataframe["failed"] = dataframe["failed"].astype("int8")

    return dataframe


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_commit_data(
    pr_name,
    build_id,
    trunk_sha,
    commit_sha,
):
    build_folder = f"{pr_name}_build{build_id}"

    commit_folder = COMMIT_DATA_DIR / build_folder

    json_files = sorted(
        commit_folder.glob(
            f"{trunk_sha}_{commit_sha}_page*.json"
        )
    )

    if not json_files:
        raise FileNotFoundError(
            f"No matching commit comparison: {build_folder}"
        )

    commits = {}
    changed_files = {}

    for path in json_files:
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source)

        for commit in data.get("commits", []):
            sha = commit.get("sha")

            if not sha:
                continue

            message = commit.get("commit", {}).get(
                "message", ""
            )

            commits[sha] = clean_text(message)

        for file in data.get("files", []):
            filename = file.get("filename")

            if not filename:
                continue

            changed_files[filename] = {
                "additions": int(file.get("additions", 0)),
                "deletions": int(file.get("deletions", 0)),
                "changes": int(file.get("changes", 0)),
                "status": file.get("status", "unknown"),
            }

    additions = sum(
        file["additions"]
        for file in changed_files.values()
    )

    deletions = sum(
        file["deletions"]
        for file in changed_files.values()
    )

    changes = sum(
        file["changes"]
        for file in changed_files.values()
    )

    return {
        "commit_message": " ".join(
            message
            for message in commits.values()
            if message
        ),
        "changed_files_text": " ".join(
            sorted(changed_files)
        ),
        "num_commits": len(commits),
        "num_changed_files": len(changed_files),
        "num_additions": additions,
        "num_deletions": deletions,
        "num_changed_lines": changes,
        "num_added_files": sum(
            file["status"] == "added"
            for file in changed_files.values()
        ),
        "num_modified_files": sum(
            file["status"] == "modified"
            for file in changed_files.values()
        ),
        "num_removed_files": sum(
            file["status"] == "removed"
            for file in changed_files.values()
        ),
    }


def normalize_dataset():
    metadata = load_metadata()

    frames = []

    total_runs = len(metadata)

    for index, row in enumerate(
        metadata.itertuples(index=False),
        start=1,
    ):
        dataframe = load_test_results(
            row.pr_name,
            row.build_id,
            row.stage_id,
        )

        commit_data = load_commit_data(
            row.pr_name,
            row.build_id,
            row.trunk_sha,
            row.build_head_sha,
        )

        actual_failed = int(dataframe["failed"].sum())

        actual_passed = len(dataframe) - actual_failed

        if actual_failed != int(row.num_fail_class):
            raise ValueError(
                f"Failure count mismatch: {row.pr_name}"
            )

        if actual_passed != int(row.num_pass_class):
            raise ValueError(
                f"Pass count mismatch: {row.pr_name}"
            )

        dataframe["project"] = PROJECT_NAME
        dataframe["pr_name"] = row.pr_name
        dataframe["build_id"] = row.build_id
        dataframe["stage_id"] = row.stage_id
        dataframe["timestamp"] = row.timestamp
        dataframe["commit_sha"] = row.build_head_sha
        dataframe["trunk_sha"] = row.trunk_sha

        dataframe["run_id"] = (
            f"{PROJECT_NAME}:"
            f"{row.pr_name}:"
            f"{row.build_id}:"
            f"{row.stage_id}"
        )

        for key, value in commit_data.items():
            dataframe[key] = value

        frames.append(dataframe)

        if index % 20 == 0 or index == total_runs:
            print(
                f"Processed suite runs: {index}/{total_runs}"
            )

    if not frames:
        raise RuntimeError("No dataset records loaded.")

    dataset = pd.concat(
        frames,
        ignore_index=True,
    )

    columns = [
        "project",
        "run_id",
        "pr_name",
        "build_id",
        "stage_id",
        "timestamp",
        "commit_sha",
        "trunk_sha",
        "test_id",
        "failed",
        "duration_seconds",
        "previous_outcome_raw",
        "commit_message",
        "changed_files_text",
        "num_commits",
        "num_changed_files",
        "num_additions",
        "num_deletions",
        "num_changed_lines",
        "num_added_files",
        "num_modified_files",
        "num_removed_files",
    ]

    dataset = dataset[columns]

    dataset = dataset.sort_values(
        ["timestamp", "run_id", "test_id"]
    ).reset_index(drop=True)

    return dataset


def validate_normalized_dataset(dataset):
    if dataset.empty:
        raise ValueError("Normalized dataset is empty.")

    if dataset.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError("Duplicate test executions found.")

    required_columns = [
        "run_id",
        "test_id",
        "timestamp",
        "failed",
        "duration_seconds",
        "commit_sha",
    ]

    if dataset[required_columns].isna().any().any():
        raise ValueError("Missing required values.")

    if not dataset["failed"].isin([0, 1]).all():
        raise ValueError("Invalid failure labels.")

    if not dataset["timestamp"].is_monotonic_increasing:
        raise ValueError("Dataset is not chronologically sorted.")

    print("Normalized dataset validation passed.")


def save_dataset(dataset):
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        NORMALIZED_DATA_FILE,
        index=False,
        encoding="utf-8",
    )

    failed = int(dataset["failed"].sum())

    passed = len(dataset) - failed

    report = {
        "project": PROJECT_NAME,
        "total_rows": len(dataset),
        "total_columns": len(dataset.columns),
        "suite_runs": int(dataset["run_id"].nunique()),
        "unique_tests": int(dataset["test_id"].nunique()),
        "unique_commits": int(dataset["commit_sha"].nunique()),
        "passed_tests": passed,
        "failed_tests": failed,
        "failure_rate_percent": round(
            failed / len(dataset) * 100,
            6,
        ),
        "missing_values": int(
            dataset.isna().sum().sum()
        ),
        "duplicate_executions": int(
            dataset.duplicated(
                ["run_id", "test_id"]
            ).sum()
        ),
        "start_date": str(dataset["timestamp"].min()),
        "end_date": str(dataset["timestamp"].max()),
        "mean_duration_seconds": float(
            dataset["duration_seconds"].mean()
        ),
        "median_duration_seconds": float(
            dataset["duration_seconds"].median()
        ),
        "empty_commit_message_rows": int(
            dataset["commit_message"].eq("").sum()
        ),
        "empty_changed_file_rows": int(
            dataset["changed_files_text"].eq("").sum()
        ),
        "file_size_mb": round(
            NORMALIZED_DATA_FILE.stat().st_size
            / (1024 ** 2),
            2,
        ),
    }

    report_path = (
        RESULTS_DIR / "normalized_dataset_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nDataset summary:")
    print(json.dumps(report, indent=4))

    print(f"\nDataset saved: {NORMALIZED_DATA_FILE}")
    print(f"Report saved: {report_path}")


def main():
    print("=" * 60)
    print("QuantCI - Dataset Normalization")
    print("=" * 60)

    dataset = normalize_dataset()

    validate_normalized_dataset(dataset)

    save_dataset(dataset)

    print("\n" + "=" * 60)
    print("DATASET NORMALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()