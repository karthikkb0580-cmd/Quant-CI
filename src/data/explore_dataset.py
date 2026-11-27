from pathlib import Path
from collections import Counter
from zipfile import ZipFile, is_zipfile

import pandas as pd

from src.config import RAW_DATA_DIR


def format_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def inspect_csv(path):
    try:
        dataframe = pd.read_csv(path, nrows=5)

        print(f"\nCSV: {path.name}")
        print(f"Columns: {len(dataframe.columns)}")
        print(dataframe.columns.tolist())
        print(dataframe.to_string(index=False, max_colwidth=60))

    except Exception as error:
        print(f"Unable to inspect CSV: {error}")


def inspect_zip(path):
    try:
        if not is_zipfile(path):
            print(f"Invalid ZIP archive: {path.name}")
            return

        with ZipFile(path) as archive:
            entries = archive.infolist()

            print(f"\nZIP: {path.name}")
            print(f"Total entries: {len(entries)}")

            total_size = sum(
                entry.file_size for entry in entries
            )

            print(
                f"Uncompressed size: {format_size(total_size)}"
            )

            print("First 20 entries:")

            for entry in entries[:20]:
                print(
                    f"  {entry.filename} "
                    f"({format_size(entry.file_size)})"
                )

    except Exception as error:
        print(f"Unable to inspect ZIP: {error}")


def main():
    print("=" * 60)
    print("QuantCI - Real Dataset Inspection")
    print("=" * 60)

    print(f"\nDataset directory: {RAW_DATA_DIR}")

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(RAW_DATA_DIR)

    files = [
        path
        for path in RAW_DATA_DIR.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.name != ".gitkeep"
    ]

    if not files:
        print("No dataset files found.")
        return

    total_size = sum(
        path.stat().st_size for path in files
    )

    print(f"Total files: {len(files)}")
    print(f"Total size: {format_size(total_size)}")

    extensions = Counter(
        path.suffix.lower() or "no extension"
        for path in files
    )

    print("\nFile formats:")

    for extension, count in extensions.most_common():
        print(f"{extension}: {count}")

    largest_files = sorted(
        files,
        key=lambda path: path.stat().st_size,
        reverse=True,
    )[:15]

    print("\nLargest files:")

    for path in largest_files:
        relative_path = path.relative_to(RAW_DATA_DIR)
        size = format_size(path.stat().st_size)

        print(f"{relative_path}: {size}")

    csv_files = [
        path for path in files
        if path.suffix.lower() == ".csv"
    ]

    zip_files = [
        path for path in files
        if path.suffix.lower() == ".zip"
    ]

    print(f"\nCSV files found: {len(csv_files)}")
    print(f"ZIP archives found: {len(zip_files)}")

    for path in csv_files[:5]:
        inspect_csv(path)

    for path in zip_files[:3]:
        inspect_zip(path)

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()