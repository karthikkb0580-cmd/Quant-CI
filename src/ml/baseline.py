import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import (
    TRAIN_CONTEXT_FILE,
    VALIDATION_CONTEXT_FILE,
    TRAIN_LABELS_FILE,
    BASELINE_REPORT_FILE,
    BASELINE_PREDICTIONS_DIR,
    BASELINE_PLOT_FILE,
    RANDOM_SEED,
)

from src.ml.metrics import evaluate_predictions


BASELINE_NAMES = [
    "majority_class",
    "random_ranking",
    "global_failure_probability",
    "historical_failure_rate",
    "smoothed_historical_probability",
]


def load_context(path):
    if not path.exists():
        raise FileNotFoundError(path)

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "run_id",
        "test_id",
        "failed",
        "historical_failure_rate",
        "smoothed_failure_rate",
        "estimated_duration_seconds",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if dataframe.empty:
        raise ValueError("Empty context dataset.")

    if dataframe.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError("Duplicate test executions.")

    if not dataframe["failed"].isin([0, 1]).all():
        raise ValueError("Invalid failure labels.")

    return dataframe


def generate_baseline_scores(
    dataframe,
    global_failure_rate,
):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    total = len(dataframe)

    return {
        "majority_class": np.zeros(
            total,
            dtype=np.float64,
        ),
        "random_ranking": rng.random(total),
        "global_failure_probability": np.full(
            total,
            global_failure_rate,
            dtype=np.float64,
        ),
        "historical_failure_rate": dataframe[
            "historical_failure_rate"
        ].to_numpy(dtype=np.float64),
        "smoothed_historical_probability": dataframe[
            "smoothed_failure_rate"
        ].to_numpy(dtype=np.float64),
    }


def save_predictions(
    dataframe,
    scores,
    partition_name,
):
    output = dataframe.copy()

    for name, values in scores.items():
        output[f"score_{name}"] = values

    path = (
        BASELINE_PREDICTIONS_DIR
        / f"{partition_name}_baseline_predictions.csv"
    )

    output.to_csv(
        path,
        index=False,
    )

    print(f"Saved predictions: {path}")


def evaluate_partition(
    name,
    dataframe,
    global_failure_rate,
):
    print(f"\nEvaluating {name}")
    print(f"Records: {len(dataframe):,}")
    print(f"Failures: {dataframe['failed'].sum()}")

    scores = generate_baseline_scores(
        dataframe,
        global_failure_rate,
    )

    results = {}

    for baseline_name, predictions in scores.items():
        metrics = evaluate_predictions(
            dataframe,
            predictions,
        )

        results[baseline_name] = metrics

        classification = metrics[
            "classification"
        ]

        ranking = metrics["ranking"]

        print(f"\n{baseline_name}")

        print(
            f"Accuracy: "
            f"{classification['accuracy']:.6f}"
        )

        print(
            f"Average Precision: "
            f"{classification['average_precision']:.6f}"
        )

        print(
            f"ROC-AUC: "
            f"{classification['roc_auc']:.6f}"
        )

        print(
            f"Recall@10%: "
            f"{ranking['top_10_percent']['recall']:.6f}"
        )

        print(
            f"Recall@20%: "
            f"{ranking['top_20_percent']['recall']:.6f}"
        )

    save_predictions(
        dataframe,
        scores,
        name,
    )

    return results


def generate_comparison_plot(results):
    names = list(results)

    values = [
        results[name]["classification"][
            "average_precision"
        ]
        for name in names
    ]

    labels = [
        name.replace("_", " ").title()
        for name in names
    ]

    plt.figure(figsize=(12, 6))

    bars = plt.barh(
        labels,
        values,
    )

    plt.xlabel("Average Precision")
    plt.title("QuantCI Validation Baseline Comparison")

    for bar, value in zip(bars, values):
        plt.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.4f}",
            va="center",
        )

    plt.xlim(
        0,
        min(1.05, max(values) * 1.25 + 0.001),
    )

    plt.tight_layout()

    plt.savefig(
        BASELINE_PLOT_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved plot: {BASELINE_PLOT_FILE}")


def generate_summary(results):
    rows = []

    for name, metrics in results.items():
        classification = metrics[
            "classification"
        ]

        ranking = metrics["ranking"]

        rows.append(
            {
                "baseline": name,
                "accuracy": classification["accuracy"],
                "precision": classification["precision"],
                "recall": classification["recall"],
                "f1_score": classification["f1_score"],
                "roc_auc": classification["roc_auc"],
                "average_precision": classification[
                    "average_precision"
                ],
                "recall_at_10_percent": ranking[
                    "top_10_percent"
                ]["recall"],
                "recall_at_20_percent": ranking[
                    "top_20_percent"
                ]["recall"],
                "recall_at_50_percent": ranking[
                    "top_50_percent"
                ]["recall"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        "average_precision",
        ascending=False,
    )


def main():
    print("=" * 60)
    print("QuantCI - Baseline Failure Prediction")
    print("=" * 60)

    BASELINE_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BASELINE_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    BASELINE_PLOT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = load_context(
        TRAIN_CONTEXT_FILE
    )

    validation = load_context(
        VALIDATION_CONTEXT_FILE
    )

    train_labels = np.load(
        TRAIN_LABELS_FILE,
        allow_pickle=False,
    )

    if not np.array_equal(
        train_labels,
        train["failed"].to_numpy(),
    ):
        raise ValueError("Training label mismatch.")

    global_failure_rate = float(
        train_labels.mean()
    )

    print(
        f"\nTraining failure probability: "
        f"{global_failure_rate:.8f}"
    )

    training_results = evaluate_partition(
        "training",
        train,
        global_failure_rate,
    )

    validation_results = evaluate_partition(
        "validation",
        validation,
        global_failure_rate,
    )

    summary = generate_summary(
        validation_results
    )

    best_baseline = str(
        summary.iloc[0]["baseline"]
    )

    report = {
        "global_training_failure_rate": (
            global_failure_rate
        ),
        "random_seed": RANDOM_SEED,
        "selection_metric": "average_precision",
        "selection_partition": "validation",
        "best_baseline": best_baseline,
        "training": training_results,
        "validation": validation_results,
        "test_evaluation": "deferred",
    }

    BASELINE_REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    summary_path = (
        BASELINE_REPORT_FILE.parent
        / "baseline_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    generate_comparison_plot(
        validation_results
    )

    print("\nValidation Baseline Comparison:")

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(f"\nBest baseline: {best_baseline}")
    print(f"Report: {BASELINE_REPORT_FILE}")
    print(f"Summary: {summary_path}")

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()