import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import (
    RANDOM_SEED,
    MAX_WORKERS,
    FEATURE_NAMES_FILE,
    BASELINE_REPORT_FILE,
    LOGISTIC_REPORT_FILE,
    LOGISTIC_PREDICTIONS_FILE,
    RANDOM_FOREST_MODEL_FILE,
    RANDOM_FOREST_REPORT_FILE,
    RANDOM_FOREST_SUMMARY_FILE,
    RANDOM_FOREST_PREDICTIONS_FILE,
    RANDOM_FOREST_IMPORTANCE_FILE,
    RANDOM_FOREST_PR_PLOT_FILE,
    RANDOM_FOREST_ROC_PLOT_FILE,
    RANDOM_FOREST_IMPORTANCE_PLOT_FILE,
    RANDOM_FOREST_COMPARISON_PLOT_FILE,
    RANDOM_FOREST_CONFIGURATIONS,
)

from src.ml.metrics import evaluate_predictions

from src.ml.logistic_regression import (
    load_training_data,
    predict_scores,
    find_best_threshold,
)


def create_model(
    n_estimators,
    max_depth,
    min_samples_leaf,
    class_weight,
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        max_features="sqrt",
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=MAX_WORKERS,
    )


def train_configuration(
    X_train,
    y_train,
    X_validation,
    context,
    configuration,
):
    (
        n_estimators,
        max_depth,
        min_samples_leaf,
        class_weight,
    ) = configuration

    weight_name = (
        "balanced"
        if class_weight == "balanced"
        else "unweighted"
    )

    name = (
        f"rf_{weight_name}"
        f"_trees_{n_estimators}"
        f"_depth_{max_depth}"
    )

    print(f"\nTraining {name}")

    model = create_model(
        n_estimators,
        max_depth,
        min_samples_leaf,
        class_weight,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter() - start
    )

    start = time.perf_counter()

    scores = predict_scores(
        model,
        X_validation,
    )

    prediction_time = (
        time.perf_counter() - start
    )

    metrics = evaluate_predictions(
        context,
        scores,
        threshold=0.5,
    )

    classification = metrics["classification"]
    ranking = metrics["ranking"]

    result = {
        "model_name": name,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": weight_name,
        "training_seconds": float(training_time),
        "prediction_seconds": float(prediction_time),
        "accuracy": classification["accuracy"],
        "precision": classification["precision"],
        "recall": classification["recall"],
        "f1_score": classification["f1_score"],
        "roc_auc": classification["roc_auc"],
        "average_precision": classification["average_precision"],
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

    print(f"Training time: {training_time:.3f} seconds")

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


def save_feature_importance(model):
    feature_names = json.loads(
        FEATURE_NAMES_FILE.read_text(
            encoding="utf-8"
        )
    )

    importance = model.feature_importances_

    if len(feature_names) != len(importance):
        raise ValueError("Feature count mismatch.")

    dataframe = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    dataframe = dataframe.sort_values(
        "importance",
        ascending=False,
    )

    dataframe.to_csv(
        RANDOM_FOREST_IMPORTANCE_FILE,
        index=False,
    )

    return dataframe


def load_logistic_scores(context):
    predictions = pd.read_csv(
        LOGISTIC_PREDICTIONS_FILE,
        low_memory=False,
    )

    keys = ["run_id", "test_id"]

    aligned = context[keys].merge(
        predictions[keys + ["failure_score"]],
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if aligned["failure_score"].isna().any():
        raise ValueError(
            "Missing Logistic Regression predictions."
        )

    return aligned["failure_score"].to_numpy()


def generate_plots(
    labels,
    context,
    scores,
    summary,
    importance,
):
    baseline_scores = context[
        "smoothed_failure_rate"
    ].to_numpy()

    logistic_scores = load_logistic_scores(
        context
    )

    models = {
        "Historical Baseline": baseline_scores,
        "Logistic Regression": logistic_scores,
        "Random Forest": scores,
    }

    plt.figure(figsize=(10, 6))

    for name, predictions in models.items():
        precision, recall, _ = precision_recall_curve(
            labels,
            predictions,
        )

        plt.plot(
            recall,
            precision,
            label=name,
        )

    plt.axhline(
        labels.mean(),
        linestyle="--",
        label="Positive Class Prevalence",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("QuantCI Precision-Recall Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RANDOM_FOREST_PR_PLOT_FILE,
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    for name, predictions in models.items():
        fpr, tpr, _ = roc_curve(
            labels,
            predictions,
        )

        plt.plot(
            fpr,
            tpr,
            label=name,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Reference",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("QuantCI ROC Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RANDOM_FOREST_ROC_PLOT_FILE,
        dpi=200,
    )

    plt.close()

    top_features = importance.head(20).sort_values(
        "importance"
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.xlabel("Impurity-Based Importance")
    plt.title("Random Forest Top 20 Features")
    plt.tight_layout()

    plt.savefig(
        RANDOM_FOREST_IMPORTANCE_PLOT_FILE,
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
    plt.title("Random Forest Configuration Comparison")
    plt.tight_layout()

    plt.savefig(
        RANDOM_FOREST_COMPARISON_PLOT_FILE,
        dpi=200,
    )

    plt.close()


def main():
    print("=" * 60)
    print("QuantCI - Random Forest Training")
    print("=" * 60)

    RANDOM_FOREST_MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RANDOM_FOREST_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RANDOM_FOREST_PR_PLOT_FILE.parent.mkdir(
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

    X_train = X_train.tocsc()

    print(f"\nTraining matrix: {X_train.shape}")
    print(f"Validation matrix: {X_validation.shape}")
    print(f"Training failures: {y_train.sum()}")
    print(f"Validation failures: {y_validation.sum()}")
    print(f"CPU workers: {MAX_WORKERS}")

    results = []

    best_model = None
    best_scores = None
    best_result = None
    best_ap = -1.0

    for configuration in RANDOM_FOREST_CONFIGURATIONS:
        model, scores, result = train_configuration(
            X_train,
            y_train,
            X_validation,
            context,
            configuration,
        )

        results.append(result)

        if result["average_precision"] > best_ap:
            best_ap = result["average_precision"]
            best_model = model
            best_scores = scores
            best_result = result

    if best_model is None:
        raise RuntimeError("No Random Forest model trained.")

    summary = pd.DataFrame(results).sort_values(
        "average_precision",
        ascending=False,
    )

    summary.to_csv(
        RANDOM_FOREST_SUMMARY_FILE,
        index=False,
    )

    threshold = find_best_threshold(
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
        threshold=threshold,
    )

    baseline_report = json.loads(
        BASELINE_REPORT_FILE.read_text(
            encoding="utf-8"
        )
    )

    logistic_report = json.loads(
        LOGISTIC_REPORT_FILE.read_text(
            encoding="utf-8"
        )
    )

    baseline_name = baseline_report["best_baseline"]

    baseline_ap = baseline_report[
        "validation"
    ][baseline_name]["classification"]["average_precision"]

    logistic_ap = logistic_report[
        "best_configuration"
    ]["average_precision"]

    comparison = {
        "historical_baseline": baseline_ap,
        "logistic_regression": logistic_ap,
        "random_forest": best_ap,
    }

    best_overall = max(
        comparison,
        key=comparison.get,
    )

    artifact = {
        "model": best_model,
        "model_name": best_result["model_name"],
        "threshold": threshold,
        "feature_count": int(X_train.shape[1]),
        "selection_metric": "average_precision",
        "class_weight": best_result["class_weight"],
    }

    joblib.dump(
        artifact,
        RANDOM_FOREST_MODEL_FILE,
        compress=3,
    )

    predictions = context.copy()

    predictions["failure_score"] = best_scores

    predictions["predicted_failure"] = (
        best_scores >= threshold
    ).astype(np.int8)

    predictions.to_csv(
        RANDOM_FOREST_PREDICTIONS_FILE,
        index=False,
    )

    importance = save_feature_importance(
        best_model
    )

    report = {
        "model_family": "random_forest",
        "configurations_trained": len(results),
        "training_rows": int(X_train.shape[0]),
        "validation_rows": int(X_validation.shape[0]),
        "features": int(X_train.shape[1]),
        "best_configuration": best_result,
        "best_threshold": threshold,
        "default_threshold_metrics": default_metrics,
        "tuned_threshold_metrics": tuned_metrics,
        "validation_comparison": comparison,
        "best_overall_by_average_precision": best_overall,
        "outperformed_baseline": best_ap > baseline_ap,
        "outperformed_logistic_regression": best_ap > logistic_ap,
        "all_configurations": results,
        "probability_calibration": "not_performed",
        "test_evaluation": "deferred",
    }

    RANDOM_FOREST_REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    generate_plots(
        y_validation,
        context,
        best_scores,
        summary,
        importance,
    )

    print("\nConfiguration Comparison:")

    print(
        summary[
            [
                "model_name",
                "average_precision",
                "roc_auc",
                "recall_at_10_percent",
                "recall_at_20_percent",
                "training_seconds",
            ]
        ].to_string(index=False)
    )

    print("\nOverall Validation Comparison:")

    for name, value in comparison.items():
        print(f"{name}: {value:.6f}")

    print(f"\nBest Random Forest: {best_result['model_name']}")
    print(f"Best threshold: {threshold:.8f}")
    print(f"Best overall by AP: {best_overall}")

    print("\nTop 10 Important Features:")

    print(
        importance.head(10).to_string(
            index=False
        )
    )

    print(f"\nSaved model: {RANDOM_FOREST_MODEL_FILE}")
    print(f"Saved report: {RANDOM_FOREST_REPORT_FILE}")

    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()