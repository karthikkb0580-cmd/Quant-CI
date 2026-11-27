import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.sparse import load_npz

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss,
)

from sklearn.calibration import calibration_curve

from src.config import (
    RANDOM_SEED,
    VALIDATION_MATRIX_FILE,
    VALIDATION_LABELS_FILE,
    VALIDATION_CONTEXT_FILE,
    TEST_MATRIX_FILE,
    TEST_LABELS_FILE,
    TEST_CONTEXT_FILE,
    BASELINE_REPORT_FILE,
    LOGISTIC_MODEL_FILE,
    LOGISTIC_REPORT_FILE,
    RANDOM_FOREST_MODEL_FILE,
    RANDOM_FOREST_REPORT_FILE,
    FINAL_MODEL_FILE,
    FINAL_MODEL_METADATA_FILE,
    FINAL_EVALUATION_REPORT,
    FINAL_EVALUATION_SUMMARY,
    FINAL_TEST_PREDICTIONS,
    FINAL_VALIDATION_PREDICTIONS,
    FINAL_PR_PLOT,
    FINAL_ROC_PLOT,
    FINAL_RECALL_PLOT,
    FINAL_AP_PLOT,
    FINAL_CONFUSION_PLOT,
    FINAL_CALIBRATION_PLOT,
)

from src.ml.metrics import evaluate_predictions


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def select_final_model(validation_scores):
    if not validation_scores:
        raise ValueError("No model candidates.")

    if not all(
        np.isfinite(value)
        for value in validation_scores.values()
    ):
        raise ValueError("Invalid validation scores.")

    return max(
        validation_scores,
        key=validation_scores.get,
    )


def load_partition(
    matrix_path,
    labels_path,
    context_path,
):
    matrix = load_npz(
        matrix_path
    ).tocsr()

    labels = np.load(
        labels_path,
        allow_pickle=False,
    )

    context = pd.read_csv(
        context_path,
        low_memory=False,
    )

    if matrix.shape[0] != len(labels):
        raise ValueError(
            "Matrix and label mismatch."
        )

    if len(context) != len(labels):
        raise ValueError(
            "Context and label mismatch."
        )

    if not np.array_equal(
        labels,
        context["failed"].to_numpy(),
    ):
        raise ValueError(
            "Context labels are not aligned."
        )

    if not np.isfinite(matrix.data).all():
        raise ValueError(
            "Invalid matrix values."
        )

    if context.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError(
            "Duplicate test executions."
        )

    if not context[
        "estimated_duration_seconds"
    ].gt(0).all():
        raise ValueError(
            "Invalid runtime estimates."
        )

    return matrix, labels, context


def load_artifacts():
    logistic = joblib.load(
        LOGISTIC_MODEL_FILE
    )

    random_forest = joblib.load(
        RANDOM_FOREST_MODEL_FILE
    )

    return {
        "logistic_regression": logistic,
        "random_forest": random_forest,
    }


def predict_model(
    artifact,
    matrix,
):
    if matrix.shape[1] != artifact[
        "feature_count"
    ]:
        raise ValueError(
            "Model feature count mismatch."
        )

    model = artifact["model"]

    positive_index = list(
        model.classes_
    ).index(1)

    scores = model.predict_proba(
        matrix
    )[:, positive_index]

    if not np.isfinite(scores).all():
        raise ValueError(
            "Non-finite prediction scores."
        )

    return scores


def generate_scores(
    matrix,
    context,
    artifacts,
    global_failure_rate,
):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    scores = {
        "majority_class": np.zeros(
            len(context)
        ),
        "random_ranking": rng.random(
            len(context)
        ),
        "global_failure_probability": np.full(
            len(context),
            global_failure_rate,
        ),
        "historical_failure_rate": context[
            "historical_failure_rate"
        ].to_numpy(),
        "smoothed_historical_probability": context[
            "smoothed_failure_rate"
        ].to_numpy(),
    }

    timings = {}

    for name, artifact in artifacts.items():
        start = time.perf_counter()

        scores[name] = predict_model(
            artifact,
            matrix,
        )

        timings[name] = (
            time.perf_counter() - start
        )

    return scores, timings


def evaluate_all(
    context,
    scores,
    artifacts,
):
    results = {}

    for name, predictions in scores.items():
        threshold = (
            artifacts[name]["threshold"]
            if name in artifacts
            else 0.5
        )

        results[name] = evaluate_predictions(
            context,
            predictions,
            threshold=threshold,
        )

    return results


def create_summary(
    results,
    timings,
):
    rows = []

    for name, metrics in results.items():
        classification = metrics[
            "classification"
        ]

        ranking = metrics["ranking"]

        rows.append(
            {
                "model": name,
                "threshold": classification[
                    "threshold"
                ],
                "accuracy": classification[
                    "accuracy"
                ],
                "precision": classification[
                    "precision"
                ],
                "recall": classification[
                    "recall"
                ],
                "f1_score": classification[
                    "f1_score"
                ],
                "roc_auc": classification[
                    "roc_auc"
                ],
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
                "failures_detected_at_20_percent": ranking[
                    "top_20_percent"
                ]["detected_failures"],
                "prediction_seconds": timings.get(
                    name,
                    None,
                ),
            }
        )

    return pd.DataFrame(rows)


def save_predictions(
    context,
    scores,
    selected_model,
    threshold,
    path,
):
    dataframe = context.copy()

    for name, values in scores.items():
        dataframe[f"score_{name}"] = values

    dataframe["failure_score"] = scores[
        selected_model
    ]

    dataframe["predicted_failure"] = (
        dataframe["failure_score"]
        >= threshold
    ).astype(np.int8)

    dataframe.to_csv(
        path,
        index=False,
    )

    return dataframe


def generate_plots(
    labels,
    scores,
    results,
    selected_model,
    threshold,
):
    comparison_models = [
        "smoothed_historical_probability",
        "logistic_regression",
        "random_forest",
    ]

    plt.figure(figsize=(10, 6))

    for name in comparison_models:
        precision, recall, _ = (
            precision_recall_curve(
                labels,
                scores[name],
            )
        )

        ap = results[name][
            "classification"
        ]["average_precision"]

        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.4f})",
        )

    plt.axhline(
        labels.mean(),
        linestyle="--",
        label="Class prevalence",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("QuantCI Held-Out Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_PR_PLOT, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))

    for name in comparison_models:
        fpr, tpr, _ = roc_curve(
            labels,
            scores[name],
        )

        auc = results[name][
            "classification"
        ]["roc_auc"]

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.4f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("QuantCI Held-Out ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_ROC_PLOT, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))

    for name in comparison_models:
        ranking = results[name]["ranking"]

        values = [
            ranking["top_10_percent"]["recall"],
            ranking["top_20_percent"]["recall"],
            ranking["top_50_percent"]["recall"],
        ]

        plt.plot(
            [10, 20, 50],
            values,
            marker="o",
            label=name,
        )

    plt.xlabel("Percentage of Tests Selected")
    plt.ylabel("Failure Recall")
    plt.title("Held-Out Test Prioritization")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_RECALL_PLOT, dpi=200)
    plt.close()

    names = list(results)

    values = [
        results[name]["classification"][
            "average_precision"
        ]
        for name in names
    ]

    plt.figure(figsize=(12, 7))

    plt.barh(
        names,
        values,
    )

    plt.xlabel("Average Precision")
    plt.title("Held-Out Model Comparison")
    plt.tight_layout()
    plt.savefig(FINAL_AP_PLOT, dpi=200)
    plt.close()

    predictions = (
        scores[selected_model] >= threshold
    ).astype(np.int8)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Passed", "Failed"],
    )

    display.plot()

    plt.title("Final Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FINAL_CONFUSION_PLOT, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))

    for name in comparison_models:
        observed, predicted = calibration_curve(
            labels,
            scores[name],
            n_bins=10,
            strategy="quantile",
        )

        plt.plot(
            predicted,
            observed,
            marker="o",
            label=name,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Mean Predicted Score")
    plt.ylabel("Observed Failure Frequency")
    plt.title("Held-Out Calibration Diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_CALIBRATION_PLOT, dpi=200)
    plt.close()


def main():
    print("=" * 60)
    print("QuantCI - Final Model Evaluation")
    print("=" * 60)

    FINAL_MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_EVALUATION_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_PR_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_report = read_json(
        BASELINE_REPORT_FILE
    )

    logistic_report = read_json(
        LOGISTIC_REPORT_FILE
    )

    forest_report = read_json(
        RANDOM_FOREST_REPORT_FILE
    )

    validation_comparison = {
        "smoothed_historical_probability": (
            baseline_report["validation"][
                "smoothed_historical_probability"
            ]["classification"]["average_precision"]
        ),
        "logistic_regression": (
            logistic_report["best_configuration"][
                "average_precision"
            ]
        ),
        "random_forest": (
            forest_report["best_configuration"][
                "average_precision"
            ]
        ),
    }

    selected_model = select_final_model(
        validation_comparison
    )

    print("\nValidation Model Comparison:")

    for name, value in validation_comparison.items():
        print(f"{name}: {value:.8f}")

    print(f"\nSelected model: {selected_model}")

    artifacts = load_artifacts()

    global_failure_rate = baseline_report[
        "global_training_failure_rate"
    ]

    if selected_model in artifacts:
        artifact = artifacts[selected_model]

        threshold = float(
            artifact["threshold"]
        )

        final_artifact = dict(artifact)

        final_artifact["model_family"] = (
            selected_model
        )

    else:
        threshold = 0.5

        final_artifact = {
            "model_family": selected_model,
            "model": None,
            "threshold": threshold,
            "feature_count": None,
        }

    final_artifact["score_type"] = (
        "uncalibrated_risk_score"
    )

    joblib.dump(
        final_artifact,
        FINAL_MODEL_FILE,
        compress=3,
    )

    metadata = {
        "selected_model": selected_model,
        "model_name": final_artifact.get(
            "model_name",
            selected_model,
        ),
        "selection_partition": "validation",
        "selection_metric": "average_precision",
        "validation_average_precision": (
            validation_comparison[selected_model]
        ),
        "threshold": threshold,
        "score_type": "uncalibrated_risk_score",
        "probability_calibration": "not_performed",
    }

    FINAL_MODEL_METADATA_FILE.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    X_validation, y_validation, validation = (
        load_partition(
            VALIDATION_MATRIX_FILE,
            VALIDATION_LABELS_FILE,
            VALIDATION_CONTEXT_FILE,
        )
    )

    validation_scores, _ = generate_scores(
        X_validation,
        validation,
        artifacts,
        global_failure_rate,
    )

    save_predictions(
        validation,
        validation_scores,
        selected_model,
        threshold,
        FINAL_VALIDATION_PREDICTIONS,
    )

    del X_validation
    del y_validation
    del validation
    del validation_scores

    X_test, y_test, context = load_partition(
        TEST_MATRIX_FILE,
        TEST_LABELS_FILE,
        TEST_CONTEXT_FILE,
    )

    print(f"\nTesting matrix: {X_test.shape}")
    print(f"Testing failures: {y_test.sum()}")
    print(f"Frozen threshold: {threshold:.8f}")

    scores, timings = generate_scores(
        X_test,
        context,
        artifacts,
        global_failure_rate,
    )

    results = evaluate_all(
        context,
        scores,
        artifacts,
    )

    summary = create_summary(
        results,
        timings,
    )

    summary.to_csv(
        FINAL_EVALUATION_SUMMARY,
        index=False,
    )

    save_predictions(
        context,
        scores,
        selected_model,
        threshold,
        FINAL_TEST_PREDICTIONS,
    )

    calibration_diagnostics = {}

    for name in [
        "smoothed_historical_probability",
        "logistic_regression",
        "random_forest",
    ]:
        calibration_diagnostics[name] = {
            "mean_score": float(
                np.mean(scores[name])
            ),
            "observed_failure_rate": float(
                np.mean(y_test)
            ),
            "brier_score": float(
                brier_score_loss(
                    y_test,
                    scores[name],
                )
            ),
        }

    report = {
        "selected_model": selected_model,
        "selection_partition": "validation",
        "selection_metric": "average_precision",
        "validation_comparison": (
            validation_comparison
        ),
        "test_rows": len(context),
        "test_failures": int(y_test.sum()),
        "test_suite_runs": int(
            context["run_id"].nunique()
        ),
        "features": int(X_test.shape[1]),
        "threshold": threshold,
        "evaluation_mode": "walk_forward",
        "model_parameters_frozen": True,
        "test_results": results,
        "selected_model_test_results": (
            results[selected_model]
        ),
        "calibration_diagnostics": (
            calibration_diagnostics
        ),
        "probability_calibration": "not_performed",
        "candidate_roster": "retrospectively_observed",
        "prediction_seconds": timings,
    }

    FINAL_EVALUATION_REPORT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    generate_plots(
        y_test,
        scores,
        results,
        selected_model,
        threshold,
    )

    print("\nFinal Testing Comparison:")

    print(
        summary[
            [
                "model",
                "average_precision",
                "roc_auc",
                "recall_at_10_percent",
                "recall_at_20_percent",
                "recall_at_50_percent",
            ]
        ].to_string(index=False)
    )

    selected_results = results[
        selected_model
    ]

    print("\nSelected Model Test Results:")

    print(
        json.dumps(
            selected_results,
            indent=4,
        )
    )

    print("\nCalibration Diagnostics:")

    print(
        json.dumps(
            calibration_diagnostics,
            indent=4,
        )
    )

    print(f"\nFinal model: {FINAL_MODEL_FILE}")
    print(f"Report: {FINAL_EVALUATION_REPORT}")
    print(f"Predictions: {FINAL_TEST_PREDICTIONS}")

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()