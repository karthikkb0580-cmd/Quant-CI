import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "results"
CALIBRATION_DIR = RESULTS_DIR / "calibration"
MODELS_DIR = BASE_DIR / "models"

VALIDATION_FILE = RESULTS_DIR / "final_validation_predictions.csv"
TEST_FILE = RESULTS_DIR / "final_test_predictions.csv"

RANDOM_SEED = 42
EPSILON = 1e-10

METHODS = [
    "uncalibrated",
    "platt",
    "isotonic",
]


def validate_inputs(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)

    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if len(scores) != len(labels):
        raise ValueError("Input lengths do not match")

    if len(scores) == 0:
        raise ValueError("Inputs cannot be empty")

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain invalid values")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid failure labels")

    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be between zero and one")

    return scores, labels.astype(int)


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(path)

    dataframe = pd.read_csv(path)

    required = {
        "run_id",
        "test_id",
        "failed",
        "failure_score",
        "timestamp",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if dataframe[list(required)].isnull().any().any():
        raise ValueError("Missing required values")

    validate_inputs(
        dataframe["failure_score"],
        dataframe["failed"],
    )

    return dataframe


def fit_calibrator(method, scores, labels):
    scores, labels = validate_inputs(scores, labels)

    if method == "uncalibrated":
        return None

    if len(np.unique(labels)) != 2:
        raise ValueError("Both outcome classes are required")

    if method == "platt":
        model = LogisticRegression(
            C=1000.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_SEED,
        )

        model.fit(scores.reshape(-1, 1), labels)

        return model

    if method == "isotonic":
        model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

        model.fit(scores, labels)

        return model

    raise ValueError(f"Unknown method: {method}")


def predict_calibrated(method, model, scores):
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1:
        raise ValueError("Scores must be one-dimensional")

    if not np.isfinite(scores).all():
        raise ValueError("Invalid scores")

    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be between zero and one")

    if method == "uncalibrated":
        probabilities = scores

    elif method == "platt":
        probabilities = model.predict_proba(
            scores.reshape(-1, 1)
        )[:, 1]

    elif method == "isotonic":
        probabilities = model.predict(scores)

    else:
        raise ValueError(f"Unknown method: {method}")

    return np.clip(
        probabilities,
        EPSILON,
        1.0 - EPSILON,
    )


def expected_calibration_error(
    labels,
    probabilities,
    bins=10,
):
    labels, probabilities = (
        np.asarray(labels),
        np.asarray(probabilities),
    )

    edges = np.linspace(0.0, 1.0, bins + 1)

    assignments = np.digitize(
        probabilities,
        edges[1:-1],
    )

    error = 0.0

    for index in range(bins):
        mask = assignments == index

        if not mask.any():
            continue

        observed = labels[mask].mean()
        predicted = probabilities[mask].mean()

        error += mask.mean() * abs(
            observed - predicted
        )

    return float(error)


def evaluate_probabilities(labels, probabilities):
    probabilities, labels = validate_inputs(
        probabilities,
        labels,
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0 - EPSILON,
    )

    return {
        "brier_score": float(
            brier_score_loss(labels, probabilities)
        ),
        "log_loss": float(
            log_loss(
                labels,
                probabilities,
                labels=[0, 1],
            )
        ),
        "expected_calibration_error": (
            expected_calibration_error(
                labels,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "mean_predicted_probability": float(
            probabilities.mean()
        ),
        "observed_failure_rate": float(
            labels.mean()
        ),
    }


def cross_validate_methods(dataframe):
    scores = dataframe["failure_score"].to_numpy()
    labels = dataframe["failed"].to_numpy()
    groups = dataframe["run_id"].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    predictions = {
        method: np.full(len(dataframe), np.nan)
        for method in METHODS
    }

    for train_indices, validation_indices in splitter.split(
        scores.reshape(-1, 1),
        labels,
        groups,
    ):
        train_scores = scores[train_indices]
        train_labels = labels[train_indices]

        if len(np.unique(train_labels)) != 2:
            raise ValueError(
                "Calibration fold lacks both classes"
            )

        for method in METHODS:
            model = fit_calibrator(
                method,
                train_scores,
                train_labels,
            )

            predictions[method][validation_indices] = (
                predict_calibrated(
                    method,
                    model,
                    scores[validation_indices],
                )
            )

    results = {}

    for method, probabilities in predictions.items():
        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"Incomplete predictions: {method}"
            )

        results[method] = evaluate_probabilities(
            labels,
            probabilities,
        )

    return results


def plot_calibration(labels, predictions):
    plt.figure(figsize=(9, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for method, probabilities in predictions.items():
        observed, predicted = calibration_curve(
            labels,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )

        plt.plot(
            predicted,
            observed,
            marker="o",
            label=method,
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed failure frequency")
    plt.title("QuantCI Probability Calibration")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        CALIBRATION_DIR / "calibration_curve.png",
        dpi=200,
    )

    plt.close()


def plot_metrics(summary):
    metrics = [
        "brier_score",
        "log_loss",
        "expected_calibration_error",
    ]

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        plt.bar(
            summary["method"],
            summary[metric],
        )

        plt.title(metric.replace("_", " ").title())
        plt.ylabel(metric)
        plt.tight_layout()

        plt.savefig(
            CALIBRATION_DIR / f"{metric}.png",
            dpi=200,
        )

        plt.close()


def main():
    started = time.perf_counter()

    CALIBRATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation = load_predictions(VALIDATION_FILE)
    test = load_predictions(TEST_FILE)

    if len(validation) != 19738:
        raise ValueError(
            "Expected corrected validation dataset"
        )

    if int(validation["failed"].sum()) != 37:
        raise ValueError(
            "Unexpected validation failure count"
        )

    if len(test) != 27786:
        raise ValueError("Unexpected test dataset size")

    if int(test["failed"].sum()) != 15:
        raise ValueError("Unexpected test failure count")

    validation_runs = set(validation["run_id"])
    test_runs = set(test["run_id"])

    if validation_runs.intersection(test_runs):
        raise ValueError(
            "Validation and test runs overlap"
        )

    validation_end = pd.to_datetime(
        validation["timestamp"],
        utc=True,
    ).max()

    test_start = pd.to_datetime(
        test["timestamp"],
        utc=True,
    ).min()

    if validation_end >= test_start:
        raise ValueError(
            "Invalid chronological partition boundary"
        )

    validation_results = cross_validate_methods(
        validation
    )

    selected_method = min(
        METHODS,
        key=lambda method: (
            validation_results[method]["brier_score"],
            validation_results[method]["log_loss"],
        ),
    )

    validation_scores = validation[
        "failure_score"
    ].to_numpy()

    validation_labels = validation[
        "failed"
    ].to_numpy()

    test_scores = test[
        "failure_score"
    ].to_numpy()

    test_labels = test[
        "failed"
    ].to_numpy()

    models = {}
    test_predictions = {}
    test_results = {}
    summary_records = []

    for method in METHODS:
        model = fit_calibrator(
            method,
            validation_scores,
            validation_labels,
        )

        models[method] = model

        probabilities = predict_calibrated(
            method,
            model,
            test_scores,
        )

        test_predictions[method] = probabilities

        test_results[method] = evaluate_probabilities(
            test_labels,
            probabilities,
        )

        summary_records.append({
            "method": method,
            **test_results[method],
        })

        test[f"probability_{method}"] = probabilities

    test["calibrated_failure_probability"] = (
        test_predictions[selected_method]
    )

    validation["calibrated_failure_probability"] = (
        predict_calibrated(
            selected_method,
            models[selected_method],
            validation_scores,
        )
    )

    summary = pd.DataFrame(summary_records)

    summary.to_csv(
        CALIBRATION_DIR / "calibration_summary.csv",
        index=False,
    )

    test.to_csv(
        CALIBRATION_DIR / "calibrated_test_predictions.csv",
        index=False,
    )

    validation.to_csv(
        CALIBRATION_DIR
        / "calibrated_validation_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "method": selected_method,
            "model": models[selected_method],
            "input_column": "failure_score",
            "output_column": (
                "calibrated_failure_probability"
            ),
        },
        MODELS_DIR / "probability_calibrator.joblib",
    )

    plot_calibration(
        test_labels,
        test_predictions,
    )

    plot_metrics(summary)

    report = {
        "dataset": "LRTS",
        "project": "activemq",
        "selected_method": selected_method,
        "selection_metric": "brier_score",
        "selection_partition": "validation",
        "validation_strategy": (
            "three_fold_stratified_group_cross_validation"
        ),
        "group_column": "run_id",
        "validation_rows": len(validation),
        "validation_failures": int(
            validation_labels.sum()
        ),
        "test_rows": len(test),
        "test_failures": int(test_labels.sum()),
        "validation_results": validation_results,
        "test_results": test_results,
        "selected_method_test_results": (
            test_results[selected_method]
        ),
        "test_data_used_for_fitting": False,
        "original_model_retrained": False,
        "original_predictions_preserved": True,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    report_path = (
        CALIBRATION_DIR / "calibration_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("=" * 70)
    print("QUANTCI PROBABILITY CALIBRATION")
    print("=" * 70)

    print("\nValidation Cross-Validation Results")

    print(
        pd.DataFrame(validation_results)
        .T.to_string()
    )

    print("\nSelected method:", selected_method)

    print("\nHeld-Out Test Results")

    print(summary.to_string(index=False))

    print("\nReport:", report_path)

    print("\nCALIBRATION COMPLETED")


if __name__ == "__main__":
    main()