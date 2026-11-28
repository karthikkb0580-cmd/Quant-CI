import json
import zipfile
from pathlib import Path
from collections import defaultdict

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"

OUTPUT_DIR = BASE_DIR / "results" / "multi_repository"

MIN_ARCHIVE_SIZE = 100 * 1024 * 1024


def discover_archives():
    return sorted(
        [
            path
            for path in RAW_DIR.rglob("*.zip")
            if path.stat().st_size >= MIN_ARCHIVE_SIZE
        ],
        key=lambda path: path.stat().st_size,
        reverse=True,
    )


def discover_repositories(archive_path):
    repositories = defaultdict(
        lambda: {
            "test_result_archives": 0,
            "commit_files": 0,
            "uncompressed_bytes": 0,
            "test_runs": set(),
            "commit_runs": set(),
        }
    )

    with zipfile.ZipFile(archive_path, "r") as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue

            parts = entry.filename.replace(
                "\\", "/"
            ).split("/")

            if "processed_test_result" in parts:
                index = parts.index(
                    "processed_test_result"
                )

                if len(parts) <= index + 2:
                    continue

                project = parts[index + 1]
                run = parts[index + 2]

                if parts[-1] == "test_class.csv.zip":
                    repositories[project][
                        "test_result_archives"
                    ] += 1

                    repositories[project][
                        "test_runs"
                    ].add(run)

                    repositories[project][
                        "uncompressed_bytes"
                    ] += entry.file_size

            if "shadata" in parts:
                index = parts.index("shadata")

                if len(parts) <= index + 3:
                    continue

                project = parts[index + 1]

                if parts[index + 2] != "compare_commits":
                    continue

                run = parts[index + 3]

                if parts[-1].endswith(".json"):
                    repositories[project][
                        "commit_files"
                    ] += 1

                    repositories[project][
                        "commit_runs"
                    ].add(run)

                    repositories[project][
                        "uncompressed_bytes"
                    ] += entry.file_size

    records = []

    for project, values in repositories.items():
        matched_runs = values[
            "test_runs"
        ].intersection(
            values["commit_runs"]
        )

        records.append({
            "repository": project,
            "test_result_archives": values[
                "test_result_archives"
            ],
            "commit_files": values[
                "commit_files"
            ],
            "test_runs": len(
                values["test_runs"]
            ),
            "commit_runs": len(
                values["commit_runs"]
            ),
            "matched_runs": len(
                matched_runs
            ),
            "uncompressed_size_mb": round(
                values["uncompressed_bytes"]
                / (1024 * 1024),
                2,
            ),
        })

    return records


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    archives = discover_archives()

    if not archives:
        raise FileNotFoundError(
            "No large ZIP archives found under data/raw"
        )

    print("=" * 70)
    print("QUANTCI MULTI-REPOSITORY DISCOVERY")
    print("=" * 70)

    all_records = []

    for archive_path in archives:
        print(f"\nInspecting: {archive_path}")

        records = discover_repositories(
            archive_path
        )

        for record in records:
            record["source_archive"] = str(
                archive_path.relative_to(BASE_DIR)
            )

        all_records.extend(records)

    if not all_records:
        raise ValueError(
            "No compatible repository structures found"
        )

    summary = pd.DataFrame(
        all_records
    ).sort_values(
        ["matched_runs", "test_result_archives"],
        ascending=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "repository_inventory.csv",
        index=False,
    )

    report = {
        "archives_inspected": len(archives),
        "repositories_discovered": int(
            summary["repository"].nunique()
        ),
        "repositories": summary.to_dict(
            orient="records"
        ),
    }

    report_path = (
        OUTPUT_DIR / "repository_inventory.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nAvailable repositories\n")

    print(
        summary.drop(
            columns=["source_archive"]
        ).to_string(index=False)
    )

    print(
        "\nRepositories discovered:",
        report["repositories_discovered"],
    )

    print("\nReport:", report_path)

    print("\nREPOSITORY DISCOVERY COMPLETED")


if __name__ == "__main__":
    main()