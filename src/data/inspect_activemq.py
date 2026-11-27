import json

import pandas as pd

from collections import Counter
from zipfile import ZipFile

from src.config import RAW_DATA_DIR

DATASET_DIR = RAW_DATA_DIR / "activemq"


def inspect_csv(path):
    dataframe = pd.read_csv(path, nrows=5)

    print(f"\nFile: {path.name}")
    print(f"Columns: {dataframe.columns.tolist()}")
    print(dataframe.to_string(index=False, max_colwidth=80))


def inspect_zip(path):
    with ZipFile(path) as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        print(f"\nArchive: {path.relative_to(DATASET_DIR)}")

        if not csv_files:
            print("No internal CSV files.")
            return

        with archive.open(csv_files[0]) as source:
            dataframe = pd.read_csv(source, nrows=5)

        print(f"Internal CSV: {csv_files[0]}")
        print(f"Columns: {dataframe.columns.tolist()}")
        print(dataframe.to_string(index=False, max_colwidth=80))


def inspect_json(path):
    with path.open("r", encoding="utf-8") as source:
        data = json.load(source)

    print(f"\nJSON: {path.relative_to(DATASET_DIR)}")
    print(f"Root type: {type(data).__name__}")

    if not isinstance(data, dict):
        return

    print(f"Root keys: {list(data.keys())}")

    commits = data.get("commits", [])
    changed_files = data.get("files", [])

    if isinstance(commits, list):
        print(f"Commits: {len(commits)}")

        if commits and isinstance(commits[0], dict):
            print(f"Commit keys: {list(commits[0].keys())}")

            commit = commits[0].get("commit", {})

            if isinstance(commit, dict):
                print(f"Nested commit keys: {list(commit.keys())}")
                print(f"Commit message: {commit.get('message')}")

    if isinstance(changed_files, list):
        print(f"Changed files: {len(changed_files)}")

        if changed_files and isinstance(changed_files[0], dict):
            print(f"File keys: {list(changed_files[0].keys())}")

        for file in changed_files[:5]:
            if not isinstance(file, dict):
                continue

            print(
                {
                    "filename": file.get("filename"),
                    "status": file.get("status"),
                    "additions": file.get("additions"),
                    "deletions": file.get("deletions"),
                    "changes": file.get("changes"),
                }
            )


def main():
    print("=" * 60)
    print("QuantCI - ActiveMQ Dataset Inspection")
    print("=" * 60)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)

    files = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.name != ".DS_Store"
    )

    if not files:
        raise RuntimeError("No extracted dataset files found.")

    print(f"\nTotal usable files: {len(files)}")

    total_size = sum(path.stat().st_size for path in files)

    print(f"Total size: {total_size / (1024 ** 2):.2f} MB")

    formats = Counter(
        path.suffix.lower()
        for path in files
    )

    print("\nFile formats:")

    for extension, count in formats.most_common():
        print(f"{extension}: {count}")

    directory_groups = Counter(
        path.relative_to(DATASET_DIR).parts[1]
        for path in files
        if len(path.relative_to(DATASET_DIR).parts) > 2
    )

    print("\nDataset categories:")

    for directory, count in directory_groups.most_common():
        print(f"{directory}: {count}")

    csv_files = [
        path for path in files
        if path.suffix.lower() == ".csv"
    ]

    zip_files = [
        path for path in files
        if path.suffix.lower() == ".zip"
    ]

    json_files = [
        path for path in files
        if path.suffix.lower() == ".json"
    ]

    test_archives = [
        path for path in zip_files
        if "processed_test_result" in str(path).lower()
    ]

    compare_files = [
        path for path in json_files
        if "compare_commits" in str(path).lower()
    ]

    print(f"\nCSV files: {len(csv_files)}")
    print(f"ZIP archives: {len(zip_files)}")
    print(f"JSON files: {len(json_files)}")
    print(f"Test result archives: {len(test_archives)}")
    print(f"Commit comparison files: {len(compare_files)}")

    print("\nCSV samples:")

    for path in csv_files[:3]:
        inspect_csv(path)

    print("\nTest result samples:")

    for path in (test_archives or zip_files)[:3]:
        inspect_zip(path)

    print("\nCode change samples:")

    for path in (compare_files or json_files)[:3]:
        inspect_json(path)

    print("\n" + "=" * 60)
    print("ACTIVEMQ INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()