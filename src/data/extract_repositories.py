import json
import shutil
import zipfile
import time
from pathlib import Path, PurePosixPath

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"

ARCHIVE_FILE = (
    RAW_DIR
    / "processed_lrts"
    / "long_running_test_suites.zip"
)

METADATA_FILE = (
    RAW_DIR
    / "LRTS"
    / "artifact"
    / "metadata"
    / "dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "multi_repository"

PROJECTS = ["karaf", "hbase"]


def identify_project(parts):
    if "processed_test_result" in parts:
        index = parts.index("processed_test_result")

        if len(parts) > index + 2:
            if parts[-1] == "test_class.csv.zip":
                return parts[index + 1]

    if "shadata" in parts:
        index = parts.index("shadata")

        if len(parts) > index + 3:
            if (
                parts[index + 2] == "compare_commits"
                and parts[-1].endswith(".json")
            ):
                return parts[index + 1]

    return None


def safe_destination(root, filename):
    normalized = filename.replace("\\", "/")

    path = PurePosixPath(normalized)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {filename}")

    destination = root.joinpath(*path.parts).resolve()

    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Invalid destination: {filename}")

    return destination


def extract_repositories():
    statistics = {
        project: {
            "extracted_files": 0,
            "test_result_archives": 0,
            "commit_files": 0,
            "extracted_bytes": 0,
        }
        for project in PROJECTS
    }

    with zipfile.ZipFile(ARCHIVE_FILE, "r") as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue

            parts = entry.filename.replace(
                "\\", "/"
            ).split("/")

            project = identify_project(parts)

            if project not in PROJECTS:
                continue

            root = RAW_DIR / project

            destination = safe_destination(
                root,
                entry.filename,
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with archive.open(entry) as source:
                with destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

            stats = statistics[project]

            stats["extracted_files"] += 1
            stats["extracted_bytes"] += entry.file_size

            if entry.filename.endswith(
                "test_class.csv.zip"
            ):
                stats["test_result_archives"] += 1
            else:
                stats["commit_files"] += 1

    return statistics


def analyze_metadata():
    dataframe = pd.read_csv(METADATA_FILE)

    records = []

    for project in PROJECTS:
        subset = dataframe[
            dataframe["project"].str.lower() == project
        ].copy()

        if subset.empty:
            raise ValueError(
                f"No metadata available for {project}"
            )

        timestamps = pd.to_datetime(
            subset["build_timestamp"],
            unit="s",
            utc=True,
            errors="coerce",
        )

        records.append({
            "project": project,
            "metadata_rows": len(subset),
            "unique_pull_requests": int(
                subset["pr_name"].nunique()
            ),
            "unique_builds": int(
                subset[
                    ["pr_name", "build_id"]
                ].drop_duplicates().shape[0]
            ),
            "reported_failed_classes": int(
                subset["num_fail_class"].fillna(0).sum()
            ),
            "reported_passed_classes": int(
                subset["num_pass_class"].fillna(0).sum()
            ),
            "missing_timestamps": int(
                timestamps.isna().sum()
            ),
            "earliest_timestamp": str(
                timestamps.min()
            ),
            "latest_timestamp": str(
                timestamps.max()
            ),
            "missing_commit_sha": int(
                subset["build_head_sha"].isna().sum()
            ),
        })

        subset.to_csv(
            OUTPUT_DIR / f"{project}_metadata.csv",
            index=False,
        )

    return records


def main():
    started = time.perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not ARCHIVE_FILE.exists():
        raise FileNotFoundError(ARCHIVE_FILE)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(METADATA_FILE)

    print("=" * 70)
    print("QUANTCI MULTI-REPOSITORY EXTRACTION")
    print("=" * 70)

    statistics = extract_repositories()

    metadata = analyze_metadata()

    records = []

    for project in PROJECTS:
        stats = statistics[project]

        records.append({
            "project": project,
            **stats,
            "extracted_size_mb": round(
                stats["extracted_bytes"]
                / (1024 * 1024),
                2,
            ),
        })

    extraction_summary = pd.DataFrame(records)

    metadata_summary = pd.DataFrame(metadata)

    extraction_summary.to_csv(
        OUTPUT_DIR / "extraction_summary.csv",
        index=False,
    )

    metadata_summary.to_csv(
        OUTPUT_DIR / "metadata_summary.csv",
        index=False,
    )

    report = {
        "selected_projects": PROJECTS,
        "source_dataset": "LRTS",
        "extraction_results": records,
        "metadata_results": metadata,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    report_path = (
        OUTPUT_DIR / "extraction_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nExtraction Summary\n")

    print(
        extraction_summary.to_string(index=False)
    )

    print("\nMetadata Summary\n")

    print(
        metadata_summary.to_string(index=False)
    )

    print("\nReport:", report_path)

    print("\nMULTI-REPOSITORY EXTRACTION COMPLETED")


if __name__ == "__main__":
    main()