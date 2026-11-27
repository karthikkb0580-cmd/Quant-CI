import os
import sys
from importlib.metadata import version, PackageNotFoundError

from src.config import (
    BASE_DIR,
    RANDOM_SEED,
    DATASET_NAME,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    MAX_WORKERS,
)

MINIMUM_PYTHON_VERSION = (3, 14)

os.environ["OMP_NUM_THREADS"] = str(MAX_WORKERS)
os.environ["MKL_NUM_THREADS"] = str(MAX_WORKERS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_WORKERS)

print("=" * 60)
print("QuantCI - Environment Verification")
print("=" * 60)

print(f"Python: {sys.version.split()[0]}")
print(f"Project: {BASE_DIR}")
print(f"Dataset: {DATASET_NAME}")
print(f"Random seed: {RANDOM_SEED}")

if sys.version_info[:2] < MINIMUM_PYTHON_VERSION:
    raise RuntimeError("Python 3.14 or newer is required.")

print("\n[PASS] Python version")

if sys.prefix == sys.base_prefix:
    print("[WARN] Virtual environment is not active.")
else:
    print("[PASS] Virtual environment")

directories = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
]

print("\nChecking directories:")

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"[PASS] {directory.relative_to(BASE_DIR)}")

packages = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "ortools",
    "qiskit",
    "qiskit-aer",
    "qiskit-algorithms",
    "matplotlib",
    "streamlit",
    "plotly",
    "pytest",
]

missing_packages = []

print("\nChecking packages:")

for package in packages:
    try:
        print(f"[PASS] {package}: {version(package)}")
    except PackageNotFoundError:
        missing_packages.append(package)
        print(f"[FAIL] {package}: Not installed")

if missing_packages:
    raise RuntimeError(
        "Missing dependencies: " + ", ".join(missing_packages)
    )

print("\n" + "=" * 60)
print("QUANTCI ENVIRONMENT READY")
print("=" * 60)