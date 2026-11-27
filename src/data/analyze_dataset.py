import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import (
    NORMALIZED_DATA_FILE,
    RESULTS_DIR,
    PLOTS_DIR,
)


def load_dataset():
    if not NORMALIZED_DATA_FILE.exists():
        raise FileNotFoundError(NORMALIZED_DATA_FILE)

    dataframe = pd.read_csv(
        NORMALIZED_DATA_FILE,
        low_memory=False,
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
        errors="raise",
    )

    return dataframe


def save_plot(filename):
    path = PLOTS_DIR / filename

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Generated: {filename}")


def plot_class_distribution(dataframe):
    counts = dataframe["failed"].value_counts().reindex(
        [0, 1],
        fill_value=0,
    )

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        ["Passed", "Failed"],
        counts.values,
        color=["#22C55E", "#EF4444"],
    )

    plt.yscale("log")
    plt.title("Test Outcome Distribution")
    plt.ylabel("Number of Executions (Log Scale)")

    for bar, count in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            max(count, 1) * 1.1,
            f"{count:,}",
            ha="center",
        )

    save_plot("class_distribution.png")


def plot_failure_percentage(dataframe):
    counts = dataframe["failed"].value_counts().reindex(
        [0, 1],
        fill_value=0,
    )

    plt.figure(figsize=(8, 6))

    plt.pie(
        counts.values,
        labels=["Passed", "Failed"],
        colors=["#22C55E", "#EF4444"],
        autopct="%1.3f%%",
        startangle=90,
        explode=[0, 0.15],
    )

    plt.title("Test Outcome Percentage")

    save_plot("failure_percentage.png")


def plot_duration_distribution(dataframe):
    durations = dataframe["duration_seconds"]

    upper_limit = durations.quantile(0.99)

    filtered = durations[durations <= upper_limit]

    plt.figure(figsize=(10, 5))

    plt.hist(
        filtered,
        bins=50,
        color="#3B82F6",
        edgecolor="white",
    )

    plt.title("Test Execution Duration Distribution")
    plt.xlabel("Duration in Seconds")
    plt.ylabel("Number of Executions")

    save_plot("duration_distribution.png")


def plot_top_failing_tests(dataframe):
    failed_tests = (
        dataframe[dataframe["failed"] == 1]
        .groupby("test_id")
        .size()
        .nlargest(15)
        .sort_values()
    )

    labels = [
        name.rsplit(".", 1)[-1]
        for name in failed_tests.index
    ]

    plt.figure(figsize=(12, 8))

    plt.barh(
        labels,
        failed_tests.values,
        color="#EF4444",
    )

    plt.title("Top 15 Frequently Failing Test Classes")
    plt.xlabel("Observed Failures")

    save_plot("top_failing_tests.png")


def plot_monthly_failure_rate(dataframe):
    monthly = (
        dataframe.set_index("timestamp")
        .resample("ME")["failed"]
        .agg(["sum", "count"])
    )

    monthly["failure_rate"] = (
        monthly["sum"]
        / monthly["count"].replace(0, np.nan)
        * 100
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        monthly.index,
        monthly["failure_rate"],
        marker="o",
        color="#8B5CF6",
    )

    plt.title("Monthly Test Failure Rate")
    plt.xlabel("Month")
    plt.ylabel("Failure Rate (%)")
    plt.grid(alpha=0.2)

    save_plot("monthly_failure_rate.png")


def plot_failures_per_build(dataframe):
    builds = (
        dataframe.groupby("run_id")["failed"]
        .sum()
    )

    plt.figure(figsize=(10, 5))

    bins = np.arange(
        -0.5,
        int(builds.max()) + 1.5,
        1,
    )

    plt.hist(
        builds,
        bins=bins,
        color="#F59E0B",
        edgecolor="white",
    )

    plt.title("Failure Distribution Across Suite Runs")
    plt.xlabel("Failed Test Classes per Run")
    plt.ylabel("Number of Suite Runs")

    save_plot("failures_per_build.png")


def plot_runtime_comparison(dataframe):
    passed = dataframe.loc[
        dataframe["failed"] == 0,
        "duration_seconds",
    ]

    failed = dataframe.loc[
        dataframe["failed"] == 1,
        "duration_seconds",
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        ["Passed", "Failed"],
        [passed.median(), failed.median()],
        color=["#22C55E", "#EF4444"],
    )

    plt.title("Median Runtime by Test Outcome")
    plt.ylabel("Duration in Seconds")

    save_plot("runtime_comparison.png")


def plot_code_changes(dataframe):
    builds = (
        dataframe.groupby(
            ["project", "pr_name", "build_id"],
            as_index=False,
        )
        .agg(
            num_changed_files=("num_changed_files", "first"),
            num_changed_lines=("num_changed_lines", "first"),
            failed_tests=("failed", "sum"),
        )
    )

    plt.figure(figsize=(10, 6))

    plt.scatter(
        builds["num_changed_lines"],
        builds["failed_tests"],
        alpha=0.6,
        color="#6366F1",
    )

    plt.xscale("symlog")

    plt.title("Changed Lines vs Failed Tests")
    plt.xlabel("Changed Lines per Build")
    plt.ylabel("Failed Test-Class Executions")

    save_plot("code_changes_vs_failures.png")


def plot_correlation_matrix(dataframe):
    columns = [
        "failed",
        "duration_seconds",
        "num_commits",
        "num_changed_files",
        "num_additions",
        "num_deletions",
        "num_changed_lines",
        "num_added_files",
        "num_modified_files",
        "num_removed_files",
    ]

    correlation = dataframe[columns].corr()

    plt.figure(figsize=(12, 10))

    matrix = plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(matrix)

    plt.xticks(
        range(len(columns)),
        columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(columns)),
        columns,
    )

    for row in range(len(columns)):
        for column in range(len(columns)):
            value = correlation.iloc[row, column]

            plt.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.title("Numerical Feature Correlation")

    save_plot("correlation_matrix.png")


def generate_test_statistics(dataframe):
    statistics = (
        dataframe.groupby("test_id")
        .agg(
            executions=("failed", "count"),
            failures=("failed", "sum"),
            mean_duration=("duration_seconds", "mean"),
            median_duration=("duration_seconds", "median"),
            max_duration=("duration_seconds", "max"),
        )
        .reset_index()
    )

    statistics["failure_rate"] = (
        statistics["failures"]
        / statistics["executions"]
    )

    statistics = statistics.sort_values(
        ["failures", "executions"],
        ascending=False,
    )

    output = RESULTS_DIR / "test_statistics.csv"

    statistics.to_csv(output, index=False)

    print(f"Generated: {output.name}")

    return statistics


def generate_build_statistics(dataframe):
    statistics = (
        dataframe.groupby("run_id")
        .agg(
            timestamp=("timestamp", "first"),
            total_tests=("failed", "count"),
            failed_tests=("failed", "sum"),
            total_duration=("duration_seconds", "sum"),
            changed_files=("num_changed_files", "first"),
            changed_lines=("num_changed_lines", "first"),
        )
        .reset_index()
    )

    statistics["passed_tests"] = (
        statistics["total_tests"]
        - statistics["failed_tests"]
    )

    statistics["failure_rate"] = (
        statistics["failed_tests"]
        / statistics["total_tests"]
    )

    statistics = statistics.sort_values("timestamp")

    output = RESULTS_DIR / "build_statistics.csv"

    statistics.to_csv(output, index=False)

    print(f"Generated: {output.name}")

    return statistics


def generate_report(
    dataframe,
    test_statistics,
    build_statistics,
):
    failed = int(dataframe["failed"].sum())
    passed = len(dataframe) - failed

    report = {
        "total_executions": len(dataframe),
        "unique_test_classes": int(
            dataframe["test_id"].nunique()
        ),
        "suite_runs": int(
            dataframe["run_id"].nunique()
        ),
        "unique_commits": int(
            dataframe["commit_sha"].nunique()
        ),
        "passed_tests": passed,
        "failed_tests": failed,
        "failure_rate_percent": round(
            failed / len(dataframe) * 100,
            6,
        ),
        "majority_baseline_accuracy": round(
            max(passed, failed) / len(dataframe),
            6,
        ),
        "class_imbalance_ratio": (
            round(passed / failed, 2)
            if failed
            else None
        ),
        "mean_duration_seconds": float(
            dataframe["duration_seconds"].mean()
        ),
        "median_duration_seconds": float(
            dataframe["duration_seconds"].median()
        ),
        "max_duration_seconds": float(
            dataframe["duration_seconds"].max()
        ),
        "total_observed_test_time_hours": float(
            dataframe["duration_seconds"].sum() / 3600
        ),
        "suite_runs_with_failures": int(
            (build_statistics["failed_tests"] > 0).sum()
        ),
        "test_classes_with_failures": int(
            (test_statistics["failures"] > 0).sum()
        ),
        "start_date": str(
            dataframe["timestamp"].min()
        ),
        "end_date": str(
            dataframe["timestamp"].max()
        ),
        "duplicate_executions": int(
            dataframe.duplicated(
                ["run_id", "test_id"]
            ).sum()
        ),
        "missing_values": {
            column: int(count)
            for column, count in
            dataframe.isna().sum().items()
            if count > 0
        },
    }

    output = RESULTS_DIR / "eda_report.json"

    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nEDA Summary:")
    print(json.dumps(report, indent=4))


def main():
    print("=" * 60)
    print("QuantCI - Exploratory Data Analysis")
    print("=" * 60)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_dataset()

    print(f"Loaded rows: {len(dataframe):,}")
    print(f"Loaded columns: {len(dataframe.columns)}")

    plot_class_distribution(dataframe)
    plot_failure_percentage(dataframe)
    plot_duration_distribution(dataframe)
    plot_top_failing_tests(dataframe)
    plot_monthly_failure_rate(dataframe)
    plot_failures_per_build(dataframe)
    plot_runtime_comparison(dataframe)
    plot_code_changes(dataframe)
    plot_correlation_matrix(dataframe)

    test_statistics = generate_test_statistics(
        dataframe
    )

    build_statistics = generate_build_statistics(
        dataframe
    )

    generate_report(
        dataframe,
        test_statistics,
        build_statistics,
    )

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()