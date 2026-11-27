from zipfile import ZipFile, is_zipfile
from collections import Counter
from src.config import RAW_DATA_DIR

DOWNLOAD_DIR = RAW_DATA_DIR / "processed_lrts"


def main():
    print("=" * 60)
    print("QuantCI - Processed Dataset Inspection")
    print("=" * 60)

    files = [
        path for path in DOWNLOAD_DIR.iterdir()
        if path.is_file()
    ]

    if not files:
        print("No downloaded dataset found.")
        return

    for path in files:
        size_gb = path.stat().st_size / (1024 ** 3)

        print(f"\nFile: {path.name}")
        print(f"Compressed size: {size_gb:.3f} GB")

        if not is_zipfile(path):
            print("Archive is not a supported ZIP file.")
            continue

        with ZipFile(path) as archive:
            entries = archive.infolist()

            total_size = sum(
                entry.file_size for entry in entries
            )

            print(f"Total entries: {len(entries)}")
            print(
                f"Uncompressed size: "
                f"{total_size / (1024 ** 3):.3f} GB"
            )

            folders = Counter(
                entry.filename.split("/")[0]
                for entry in entries
            )

            print("\nTop-level archive contents:")

            for folder, count in folders.most_common(20):
                print(f"{folder}: {count}")

            activemq_files = [
                entry for entry in entries
                if "activemq" in entry.filename.lower()
                and not entry.is_dir()
            ]

            activemq_size = sum(
                entry.file_size
                for entry in activemq_files
            )

            print("\nActiveMQ subset:")
            print(f"Files: {len(activemq_files)}")
            print(
                f"Size: "
                f"{activemq_size / (1024 ** 2):.2f} MB"
            )

            print("\nFirst 20 ActiveMQ files:")

            for entry in activemq_files[:20]:
                print(entry.filename)

    print("\n" + "=" * 60)
    print("ARCHIVE INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()