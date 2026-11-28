import json
import re
import time
import zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_repository"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "multi_repository"
)

METADATA_FILE = (
    RAW_DIR
    / "LRTS"
    / "artifact"
    / "metadata"
    / "dataset.csv"
)

PROJECTS = ["karaf", "hbase"]

RUN_PATTERN = re.compile(
    r"^(PR-\d+)_build(\d+)$"
)


def load_metadata(project):
    dataframe = pd.read_csv(METADATA_FILE)

    dataframe = dataframe[
        dataframe["project"].str.lower() == project
    ].copy()

    dataframe["build_id"] = (
        dataframe["build_id"].astype(int)
    )

    dataframe["stage_id"] = (
        dataframe["stage_id"].astype(str)
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["build_timestamp"],
        unit="s",
        utc=True,
        errors="coerce",
    )

    dataframe["build_duration"] = pd.to_numeric(
        dataframe["build_duration"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "timestamp",
            "build_head_sha",
            "trunk_sha",
            "build_duration",
        ]
    )

    dataframe = dataframe[
        dataframe["build_duration"] >= 0
    ].copy()

    dataframe["completion_timestamp"] = (
        dataframe["timestamp"]
        + pd.to_timedelta(
            dataframe["build_duration"],
            unit="s",
        )
    )

    keys = [
        "pr_name",
        "build_id",
        "stage_id",
    ]

    dataframe = dataframe.drop_duplicates()

    ambiguous = dataframe.duplicated(
        keys,
        keep=False,
    )

    dataframe = dataframe[
        ~ambiguous
    ].copy()

    return dataframe.set_index(keys)


def parse_archive_path(path):
    run_name = path.parent.parent.name

    stage_name = path.parent.name

    match = RUN_PATTERN.fullmatch(
        run_name
    )

    if not match:
        raise ValueError(
            f"Invalid run path: {path}"
        )

    if not stage_name.startswith("stage_"):
        raise ValueError(
            f"Invalid stage path: {path}"
        )

    return {
        "pr_name": match.group(1),
        "build_id": int(match.group(2)),
        "stage_id": stage_name.removeprefix(
            "stage_"
        ),
        "run_name": run_name,
    }


def read_test_archive(path):
    with zipfile.ZipFile(path) as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Unexpected archive contents: {path}"
            )

        with archive.open(csv_files[0]) as file:
            dataframe = pd.read_csv(file)

    required = [
        "testclass",
        "duration",
        "outcome",
        "last_outcome",
    ]

    missing = set(required).difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    dataframe = dataframe[required].copy()

    dataframe["duration"] = pd.to_numeric(
        dataframe["duration"],
        errors="coerce",
    )

    dataframe["outcome"] = pd.to_numeric(
        dataframe["outcome"],
        errors="coerce",
    )

    valid = (
        dataframe["testclass"].notna()
        & dataframe["duration"].notna()
        & np.isfinite(dataframe["duration"])
        & (dataframe["duration"] >= 0)
        & dataframe["outcome"].isin([0, 1])
    )

    invalid_rows = int(
        (~valid).sum()
    )

    dataframe = dataframe[valid].copy()

    dataframe["testclass"] = (
        dataframe["testclass"]
        .astype(str)
        .str.strip()
    )

    empty_names = dataframe["testclass"].eq("")

    invalid_rows += int(empty_names.sum())

    dataframe = dataframe[~empty_names].copy()

    dataframe = dataframe.drop_duplicates()

    duplicates = dataframe.duplicated(
        ["testclass"],
        keep=False,
    )

    ambiguous_rows = int(
        duplicates.sum()
    )

    dataframe = dataframe[
        ~duplicates
    ].copy()

    dataframe["outcome"] = (
        dataframe["outcome"].astype(int)
    )

    return dataframe, invalid_rows, ambiguous_rows


def find_commit_files(
    project,
    run_name,
    trunk_sha,
    head_sha,
):
    root = (
        RAW_DIR
        / project
        / "long_running_test_suites"
        / "shadata"
        / project
        / "compare_commits"
        / run_name
    )

    if not root.exists():
        return []

    pattern = (
        f"{trunk_sha}_{head_sha}_page*.json"
    )

    files = list(root.glob(pattern))

    def page_number(path):
        match = re.search(
            r"_page(\d+)\.json$",
            path.name,
        )

        return int(match.group(1)) if match else 0

    return sorted(
        files,
        key=page_number,
    )


def read_commit_features(paths):
    commits = {}
    changed_files = {}

    for path in paths:
        with path.open(
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        for commit in data.get("commits", []):
            sha = commit.get("sha")

            if sha:
                commits[sha] = commit

        for changed in data.get("files", []):
            filename = changed.get("filename")

            if filename:
                changed_files[filename] = changed

    messages = [
        str(
            commit.get("commit", {}).get(
                "message",
                "",
            )
        )
        for commit in commits.values()
    ]

    files = list(changed_files.values())

    statuses = Counter(
        file.get("status", "")
        for file in files
    )

    additions = sum(
        int(file.get("additions", 0) or 0)
        for file in files
    )

    deletions = sum(
        int(file.get("deletions", 0) or 0)
        for file in files
    )

    changes = sum(
        int(file.get("changes", 0) or 0)
        for file in files
    )

    return {
        "commit_message": " ".join(messages),
        "changed_files_text": " ".join(
            changed_files.keys()
        ),
        "num_commits": len(commits),
        "num_changed_files": len(files),
        "num_additions": additions,
        "num_deletions": deletions,
        "num_changed_lines": changes,
        "num_added_files": statuses["added"],
        "num_modified_files": statuses["modified"],
        "num_removed_files": statuses["removed"],
    }


def normalize_project(project):
    started = time.perf_counter()

    metadata = load_metadata(project)

    root = (
        RAW_DIR
        / project
        / "long_running_test_suites"
        / "processed_test_result"
        / project
    )

    archives = sorted(
        root.rglob("test_class.csv.zip")
    )

    if not archives:
        raise FileNotFoundError(root)

    frames = []

    counters = Counter()

    excluded = []

    commit_cache = {}

    counters["total_archives"] = len(
        archives
    )

    for index, archive in enumerate(
        archives,
        start=1,
    ):
        try:
            identity = parse_archive_path(
                archive
            )

            key = (
                identity["pr_name"],
                identity["build_id"],
                identity["stage_id"],
            )

            if key not in metadata.index:
                counters["missing_metadata"] += 1

                excluded.append({
                    "archive": str(
                        archive.relative_to(BASE_DIR)
                    ),
                    "reason": "missing_metadata",
                })

                continue

            row = metadata.loc[key]

            trunk_sha = str(
                row["trunk_sha"]
            )

            head_sha = str(
                row["build_head_sha"]
            )

            cache_key = (
                identity["run_name"],
                trunk_sha,
                head_sha,
            )

            if cache_key not in commit_cache:
                commit_files = find_commit_files(
                    project,
                    identity["run_name"],
                    trunk_sha,
                    head_sha,
                )

                if not commit_files:
                    commit_cache[cache_key] = None
                else:
                    commit_cache[cache_key] = (
                        read_commit_features(
                            commit_files
                        )
                    )

            features = commit_cache[cache_key]

            if features is None:
                counters["missing_commit_data"] += 1

                excluded.append({
                    "archive": str(
                        archive.relative_to(BASE_DIR)
                    ),
                    "reason": "missing_commit_data",
                })

                continue

            tests, invalid, ambiguous = (
                read_test_archive(archive)
            )

            counters["invalid_test_rows"] += invalid

            counters[
                "ambiguous_test_rows"
            ] += ambiguous

            if tests.empty:
                counters["empty_archives"] += 1
                continue

            tests = tests.rename(
                columns={
                    "testclass": "test_id",
                    "duration": "duration_seconds",
                    "outcome": "failed",
                    "last_outcome": "previous_outcome_raw",
                }
            )

            pr_name = identity["pr_name"]

            build_id = identity["build_id"]

            stage_id = identity["stage_id"]

            build_key = (
                f"{project}:{pr_name}:{build_id}"
            )

            run_id = (
                f"{build_key}:{stage_id}"
            )

            tests["project"] = project

            tests["run_id"] = run_id

            tests["build_key"] = build_key

            tests["pr_name"] = pr_name

            tests["build_id"] = build_id

            tests["stage_id"] = stage_id

            tests["timestamp"] = row["timestamp"]

            tests["completion_timestamp"] = (
                row["completion_timestamp"]
            )

            tests["commit_sha"] = head_sha

            tests["trunk_sha"] = trunk_sha

            for name, value in features.items():
                tests[name] = value

            frames.append(tests)

            counters["processed_archives"] += 1

        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            zipfile.BadZipFile,
        ) as error:
            counters["invalid_archives"] += 1

            excluded.append({
                "archive": str(
                    archive.relative_to(BASE_DIR)
                ),
                "reason": str(error),
            })

        if index % 100 == 0:
            print(
                f"{project}: "
                f"{index}/{len(archives)} archives"
            )

    if not frames:
        raise ValueError(
            f"No usable records for {project}"
        )

    dataframe = pd.concat(
        frames,
        ignore_index=True,
    )

    duplicate_count = int(
        dataframe.duplicated(
            ["run_id", "test_id"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Duplicate executions: {duplicate_count}"
        )

    dataframe = dataframe.sort_values(
        [
            "timestamp",
            "build_key",
            "stage_id",
            "test_id",
        ]
    ).reset_index(drop=True)

    output_file = (
        PROCESSED_DIR
        / f"{project}_normalized.csv"
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    pd.DataFrame(
        excluded,
        columns=["archive", "reason"],
    ).to_csv(
        RESULTS_DIR
        / f"{project}_excluded_archives.csv",
        index=False,
    )

    failures = int(
        dataframe["failed"].sum()
    )

    report = {
        "project": project,
        "metadata_records": len(metadata),
        **dict(counters),
        "normalized_rows": len(dataframe),
        "passed": len(dataframe) - failures,
        "failed": failures,
        "failure_rate": failures / len(dataframe),
        "unique_tests": int(
            dataframe["test_id"].nunique()
        ),
        "unique_runs": int(
            dataframe["run_id"].nunique()
        ),
        "unique_builds": int(
            dataframe["build_key"].nunique()
        ),
        "unique_commits": int(
            dataframe["commit_sha"].nunique()
        ),
        "failure_runs": int(
            dataframe.groupby("run_id")[
                "failed"
            ].max().sum()
        ),
        "earliest_timestamp": str(
            dataframe["timestamp"].min()
        ),
        "latest_timestamp": str(
            dataframe["timestamp"].max()
        ),
        "mean_duration_seconds": float(
            dataframe["duration_seconds"].mean()
        ),
        "duplicate_executions": duplicate_count,
        "output_file": str(
            output_file.relative_to(BASE_DIR)
        ),
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    print(
        json.dumps(report, indent=4)
    )

    del dataframe
    del frames

    return report


def main():
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = []

    print("=" * 70)
    print("QUANTCI MULTI-REPOSITORY NORMALIZATION")
    print("=" * 70)

    for project in PROJECTS:
        print(f"\nProcessing {project}")

        reports.append(
            normalize_project(project)
        )

    summary = pd.DataFrame(reports)

    summary.to_csv(
        RESULTS_DIR / "normalization_summary.csv",
        index=False,
    )

    report = {
        "dataset": "LRTS",
        "projects": PROJECTS,
        "total_normalized_rows": sum(
            item["normalized_rows"]
            for item in reports
        ),
        "total_failures": sum(
            item["failed"]
            for item in reports
        ),
        "repositories": reports,
    }

    report_path = (
        RESULTS_DIR
        / "normalization_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nNormalization Summary\n")

    print(
        summary[
            [
                "project",
                "normalized_rows",
                "failed",
                "failure_rate",
                "unique_runs",
                "unique_builds",
                "unique_tests",
            ]
        ].to_string(index=False)
    )

    print(
        "\nTotal normalized rows:",
        report["total_normalized_rows"],
    )

    print(
        "Total failures:",
        report["total_failures"],
    )

    print("\nReport:", report_path)

    print("\nMULTI-REPOSITORY NORMALIZATION COMPLETED")


if __name__ == "__main__":
    main()