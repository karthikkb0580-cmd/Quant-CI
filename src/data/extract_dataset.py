import shutil

from pathlib import Path
from zipfile import ZipFile, is_zipfile

from src.config import RAW_DATA_DIR

DOWNLOAD_DIR = RAW_DATA_DIR / "processed_lrts"
OUTPUT_DIR = RAW_DATA_DIR / "activemq"
PROJECT_NAME = "activemq"


def format_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def find_archives():
    if not DOWNLOAD_DIR.exists():
        raise FileNotFoundError(DOWNLOAD_DIR)

    archives = [
        path for path in DOWNLOAD_DIR.iterdir()
        if path.is_file() and is_zipfile(path)
    ]

    if not archives:
        raise FileNotFoundError(
            "No valid ZIP archives found."
        )

    return archives


def extract_archive(archive_path):
    print(f"\nArchive: {archive_path.name}")

    with ZipFile(archive_path) as archive:
        selected = [
            entry for entry in archive.infolist()
            if not entry.is_dir()
            and PROJECT_NAME in [
                part.lower()
                for part in Path(entry.filename).parts
            ]
        ]

        if not selected:
            print("No ActiveMQ files found.")
            return 0

        required_space = sum(
            entry.file_size for entry in selected
        )

        free_space = shutil.disk_usage(OUTPUT_DIR).free

        print(f"Selected files: {len(selected)}")
        print(f"Required space: {format_size(required_space)}")
        print(f"Available space: {format_size(free_space)}")

        if required_space > free_space * 0.8:
            raise RuntimeError(
                "Insufficient safe extraction space."
            )

        extracted = 0

        for entry in selected:
            destination = (
                OUTPUT_DIR / entry.filename
            ).resolve()

            if not destination.is_relative_to(
                OUTPUT_DIR.resolve()
            ):
                raise ValueError(
                    f"Unsafe archive path: {entry.filename}"
                )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with archive.open(entry) as source:
                with destination.open("wb") as target:
                    shutil.copyfileobj(
                        source,
                        target,
                        length=1024 * 1024,
                    )

            extracted += 1

            if extracted % 100 == 0:
                print(
                    f"Extracted {extracted}/{len(selected)}"
                )

        print(f"Extracted files: {extracted}")

        return extracted


def main():
    print("=" * 60)
    print("QuantCI - ActiveMQ Dataset Extraction")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    archives = find_archives()

    total_extracted = 0

    for archive in archives:
        total_extracted += extract_archive(archive)

    if total_extracted == 0:
        raise RuntimeError(
            "No ActiveMQ files extracted. "
            "Inspect the archive directory structure."
        )

    print(f"\nTotal extracted files: {total_extracted}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("ACTIVEMQ EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()