import os
import sys
from importlib.metadata import version

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

if sys.version_info[:3] != (3, 14, 10):
    raise RuntimeError("Python 3.14.10 is required.")

if sys.prefix == sys.base_prefix:
    raise RuntimeError("Virtual environment is not active.")

print("\n[PASS] Python version")
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
    if not directory.is_dir():
        raise FileNotFoundError(directory)

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
    "pytest",
]

print("\nChecking packages:")

for package in packages:
    print(f"[PASS] {package}: {version(package)}")

print("\n" + "=" * 60)
print("QUANTCI ENVIRONMENT READY")
print("=" * 60)