import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)

from src.config import (
    TRAIN_MATRIX_FILE,
    VALIDATION_MATRIX_FILE,
    TRAIN_LABELS_FILE,
    VALIDATION_LABELS_FILE,
    VALIDATION_CONTEXT_FILE,
    FEATURE_NAMES_FILE,
    BASELINE_REPORT_FILE,
    LOGISTIC_MODEL_FILE,
    LOGISTIC_REPORT_FILE,
    LOGISTIC_SUMMARY_FILE,
    LOGISTIC_PREDICTIONS_FILE,
    LOGISTIC_COEFFICIENTS_FILE,
    LOGISTIC_PR_PLOT_FILE,
    LOGISTIC_ROC_PLOT_FILE,
    LOGISTIC_COMPARISON_PLOT_FILE,
    LOGISTIC_C_VALUES,
    LOGISTIC_MAX_ITERATIONS,
    LOGISTIC_TOLERANCE,
    RANDOM_SEED,
)

from src.ml.metrics import evaluate_predictions


def load_training_data():
    X_train = load_npz(
        TRAIN_MATRIX_FILE
    ).tocsr()

    X_validation = load_npz(
        VALIDATION_MATRIX_FILE
    ).tocsr()

    y_train = np.load(
        TRAIN_LABELS_FILE,
        allow_pickle=False,
    )

    y_validation = np.load(
        VALIDATION_LABELS_FILE,
        allow_pickle=False,
    )

    context = pd.read_csv(
        VALIDATION_CONTEXT_FILE,
        low_memory=False,
    )

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "Training matrix and label mismatch."
        )

    if X_validation.shape[0] != len(y_validation):
        raise ValueError(
            "Validation matrix and label mismatch."
        )

    if X_train.shape[1] != X_validation.shape[1]:
        raise ValueError(
            "Feature count mismatch."
        )

    if not np.array_equal(
        y_validation,
        context["failed"].to_numpy(),
    ):
        raise ValueError(
            "Validation context mismatch."
        )

    if not np.isfinite(X_train.data).all():
        raise ValueError(
            "Invalid training matrix."
        )

    if not np.isfinite(X_validation.data).all():
        raise ValueError(
            "Invalid validation matrix."
        )

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError(
            "Training requires both classes."
        )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        context,
    )


def create_model(C, class_weight):
    return LogisticRegression(
        C=C,
        solver="liblinear",
        class_weight=class_weight,
        max_iter=LOGISTIC_MAX_ITERATIONS,
        tol=LOGISTIC_TOLERANCE,
        random_state=RANDOM_SEED,
    )


def predict_scores(model, matrix):
    scores = model.predict_proba(
        matrix
    )[:, 1]

    if not np.isfinite(scores).all():
        raise ValueError(
            "Invalid prediction scores."
        )

    if not np.all(
        (scores >= 0) & (scores <= 1)
    ):
        raise ValueError(
            "Prediction scores outside valid range."
        )

    return scores


def find_best_threshold(labels, scores):
    precision, recall, thresholds = (
        precision_recall_curve(
            labels,
            scores,
        )
    )

    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    index = int(np.argmax(f1))

    return float(thresholds[index])


def train_configuration(
    X_train,
    y_train,
    X_validation,
    context,
    C,
    class_weight,
):
    weight_name = (
        "balanced"
        if class_weight == "balanced"
        else "unweighted"
    )

    name = f"lr_{weight_name}_c_{C}"

    print(f"\nTraining {name}")

    model = create_model(
        C,
        class_weight,
    )

    start = time.perf_counter()

    with warnings.catch_warnings(
        record=True
    ) as captured:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        model.fit(
            X_train,
            y_train,
        )

    training_time = (
        time.perf_counter() - start
    )

    converged = not any(
        issubclass(
            item.category,
            ConvergenceWarning,
        )
        for item in captured
    )

    prediction_start = (
        time.perf_counter()
    )

    scores = predict_scores(
        model,
        X_validation,
    )

    prediction_time = (
        time.perf_counter()
        - prediction_start
    )

    metrics = evaluate_predictions(
        context,
        scores,
        threshold=0.5,
    )

    classification = metrics[
        "classification"
    ]

    ranking = metrics["ranking"]

    result = {
        "model_name": name,
        "C": C,
        "class_weight": weight_name,
        "converged": converged,
        "iterations": int(
            np.max(model.n_iter_)
        ),
        "training_seconds": float(
            training_time
        ),
        "prediction_seconds": float(
            prediction_time
        ),
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
    }

    print(
        f"Training time: "
        f"{training_time:.3f} seconds"
    )

    print(
        f"Converged: {converged}"
    )

    print(
        f"Average Precision: "
        f"{result['average_precision']:.6f}"
    )

    print(
        f"ROC-AUC: "
        f"{result['roc_auc']:.6f}"
    )

    print(
        f"Recall@10%: "
        f"{result['recall_at_10_percent']:.6f}"
    )

    print(
        f"Recall@20%: "
        f"{result['recall_at_20_percent']:.6f}"
    )

    return model, scores, result


def save_coefficients(model):
    feature_names = json.loads(
        FEATURE_NAMES_FILE.read_text(
            encoding="utf-8"
        )
    )

    coefficients = model.coef_.ravel()

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Coefficient and feature mismatch."
        )

    dataframe = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(
                coefficients
            ),
        }
    )

    dataframe = dataframe.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    dataframe.to_csv(
        LOGISTIC_COEFFICIENTS_FILE,
        index=False,
    )

    return dataframe


def generate_plots(
    labels,
    scores,
    context,
    summary,
):
    baseline_scores = context[
        "smoothed_failure_rate"
    ].to_numpy()

    precision, recall, _ = (
        precision_recall_curve(
            labels,
            scores,
        )
    )

    baseline_precision, baseline_recall, _ = (
        precision_recall_curve(
            labels,
            baseline_scores,
        )
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        recall,
        precision,
        label="Logistic Regression",
    )

    plt.plot(
        baseline_recall,
        baseline_precision,
        label="Historical Baseline",
    )

    plt.axhline(
        labels.mean(),
        linestyle="--",
        label="Positive Class Prevalence",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("QuantCI Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        LOGISTIC_PR_PLOT_FILE,
        dpi=200,
    )

    plt.close()

    fpr, tpr, _ = roc_curve(
        labels,
        scores,
    )

    baseline_fpr, baseline_tpr, _ = (
        roc_curve(
            labels,
            baseline_scores,
        )
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        fpr,
        tpr,
        label="Logistic Regression",
    )

    plt.plot(
        baseline_fpr,
        baseline_tpr,
        label="Historical Baseline",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Reference",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("QuantCI ROC Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        LOGISTIC_ROC_PLOT_FILE,
        dpi=200,
    )

    plt.close()

    ordered = summary.sort_values(
        "average_precision"
    )

    plt.figure(figsize=(12, 7))

    plt.barh(
        ordered["model_name"],
        ordered["average_precision"],
    )

    plt.xlabel("Average Precision")
    plt.title(
        "Logistic Regression Configuration Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        LOGISTIC_COMPARISON_PLOT_FILE,
        dpi=200,
    )

    plt.close()


def main():
    print("=" * 60)
    print("QuantCI - Logistic Regression Training")
    print("=" * 60)

    LOGISTIC_MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGISTIC_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGISTIC_PR_PLOT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        context,
    ) = load_training_data()

    print(f"\nTraining matrix: {X_train.shape}")

    print(
        f"Validation matrix: {X_validation.shape}"
    )

    print(
        f"Training failures: {y_train.sum()}"
    )

    print(
        f"Validation failures: {y_validation.sum()}"
    )

    results = []

    best_model = None
    best_scores = None
    best_result = None
    best_ap = -1.0

    for class_weight in [None, "balanced"]:
        for C in LOGISTIC_C_VALUES:
            model, scores, result = (
                train_configuration(
                    X_train,
                    y_train,
                    X_validation,
                    context,
                    C,
                    class_weight,
                )
            )

            results.append(result)

            if (
                result["converged"]
                and result["average_precision"] > best_ap
            ):
                best_ap = result[
                    "average_precision"
                ]

                best_model = model
                best_scores = scores
                best_result = result

    if best_model is None:
        raise RuntimeError(
            "No converged model was trained."
        )

    summary = pd.DataFrame(
        results
    ).sort_values(
        "average_precision",
        ascending=False,
    )

    summary.to_csv(
        LOGISTIC_SUMMARY_FILE,
        index=False,
    )

    best_threshold = find_best_threshold(
        y_validation,
        best_scores,
    )

    default_metrics = evaluate_predictions(
        context,
        best_scores,
        threshold=0.5,
    )

    tuned_metrics = evaluate_predictions(
        context,
        best_scores,
        threshold=best_threshold,
    )

    baseline_report = json.loads(
        BASELINE_REPORT_FILE.read_text(
            encoding="utf-8"
        )
    )

    baseline_name = baseline_report[
        "best_baseline"
    ]

    baseline_metrics = baseline_report[
        "validation"
    ][baseline_name]

    baseline_ap = baseline_metrics[
        "classification"
    ]["average_precision"]

    improvement = (
        (best_ap - baseline_ap)
        / baseline_ap
        * 100
        if baseline_ap > 0
        else None
    )

    artifact = {
        "model": best_model,
        "model_name": best_result[
            "model_name"
        ],
        "threshold": best_threshold,
        "feature_count": int(
            X_train.shape[1]
        ),
        "selection_metric": (
            "average_precision"
        ),
        "class_weight": best_result[
            "class_weight"
        ],
    }

    joblib.dump(
        artifact,
        LOGISTIC_MODEL_FILE,
        compress=3,
    )

    predictions = context.copy()

    predictions["failure_score"] = (
        best_scores
    )

    predictions["predicted_failure"] = (
        best_scores >= best_threshold
    ).astype(np.int8)

    predictions.to_csv(
        LOGISTIC_PREDICTIONS_FILE,
        index=False,
    )

    coefficients = save_coefficients(
        best_model
    )

    report = {
        "model_family": "logistic_regression",
        "configurations_trained": len(results),
        "training_rows": int(
            X_train.shape[0]
        ),
        "validation_rows": int(
            X_validation.shape[0]
        ),
        "features": int(
            X_train.shape[1]
        ),
        "best_configuration": best_result,
        "best_threshold": best_threshold,
        "default_threshold_metrics": (
            default_metrics
        ),
        "tuned_threshold_metrics": (
            tuned_metrics
        ),
        "baseline_name": baseline_name,
        "baseline_average_precision": (
            baseline_ap
        ),
        "average_precision_improvement_percent": (
            improvement
        ),
        "outperformed_baseline": (
            best_ap > baseline_ap
        ),
        "all_configurations": results,
        "test_evaluation": "deferred",
        "probability_calibration": "not_performed",
    }

    LOGISTIC_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    generate_plots(
        y_validation,
        best_scores,
        context,
        summary,
    )

    print("\nConfiguration Comparison:")

    print(
        summary[
            [
                "model_name",
                "converged",
                "average_precision",
                "roc_auc",
                "recall_at_10_percent",
                "recall_at_20_percent",
                "training_seconds",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nBest model: "
        f"{best_result['model_name']}"
    )

    print(
        f"Best threshold: "
        f"{best_threshold:.8f}"
    )

    print(
        f"Baseline AP: "
        f"{baseline_ap:.6f}"
    )

    print(
        f"Logistic Regression AP: "
        f"{best_ap:.6f}"
    )

    print(
        f"Outperformed baseline: "
        f"{best_ap > baseline_ap}"
    )

    print("\nTop 10 coefficient magnitudes:")

    print(
        coefficients.head(10).to_string(
            index=False
        )
    )

    print(
        f"\nSaved model: "
        f"{LOGISTIC_MODEL_FILE}"
    )

    print(
        f"Saved report: "
        f"{LOGISTIC_REPORT_FILE}"
    )

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()