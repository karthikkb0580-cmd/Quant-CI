import gc
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parents[2]

MATRIX_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "multi_repository"
    / "matrices"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "multi_repository"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "multi_repository"
    / "training"
)

PROJECTS = ["karaf", "hbase"]

RANDOM_SEED = 42

MAX_WORKERS = 2

TOP_FRACTIONS = [0.10, 0.20, 0.50]

LOGISTIC_CONFIGS = [
    {
        "C": c,
        "class_weight": weight,
    }
    for c in [0.01, 0.1, 1.0, 10.0]
    for weight in [None, "balanced"]
]

FOREST_CONFIGS = [
    {
        "n_estimators": trees,
        "max_depth": depth,
        "min_samples_leaf": leaf,
        "class_weight": weight,
    }
    for trees, depth, leaf in [
        (100, 8, 5),
        (100, 12, 5),
        (150, 16, 5),
    ]
    for weight in [None, "balanced"]
]


def safe_divide(numerator, denominator):
    return (
        numerator / denominator
        if denominator
        else 0.0
    )


def validate_partition(matrix, labels, context):
    if not sparse.issparse(matrix):
        raise ValueError("Expected sparse matrix")

    if matrix.shape[0] == 0:
        raise ValueError("Empty matrix")

    if matrix.shape[0] != len(labels):
        raise ValueError("Matrix-label mismatch")

    if len(labels) != len(context):
        raise ValueError("Label-context mismatch")

    if not np.isfinite(matrix.data).all():
        raise ValueError("Invalid matrix values")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid labels")

    if not np.array_equal(
        labels,
        context["failed"].to_numpy(),
    ):
        raise ValueError("Context labels do not match")

    if context.duplicated(
        ["run_id", "test_id"]
    ).any():
        raise ValueError("Duplicate executions")

    return True


def load_partition(project, partition):
    directory = MATRIX_DIR / project

    matrix = sparse.load_npz(
        directory / f"X_{partition}.npz"
    ).tocsr()

    labels = np.load(
        directory / f"y_{partition}.npy",
        allow_pickle=False,
    )

    context = pd.read_csv(
        directory / f"{partition}_context.csv"
    )

    validate_partition(
        matrix,
        labels,
        context,
    )

    return matrix, labels, context


def validate_scores(labels, scores):
    labels = np.asarray(labels)

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(labels) != len(scores):
        raise ValueError("Score length mismatch")

    if len(labels) == 0:
        raise ValueError("Empty scores")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid labels")

    if not np.isfinite(scores).all():
        raise ValueError("Invalid scores")

    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores outside valid range")

    return labels, scores


def ranking_metrics(context, scores, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("Invalid ranking fraction")

    labels, scores = validate_scores(
        context["failed"].to_numpy(),
        scores,
    )

    dataframe = context[
        ["run_id", "test_id", "failed"]
    ].copy()

    dataframe["score"] = scores

    selected = 0

    detected = 0

    failure_runs = 0

    detected_runs = 0

    for _, group in dataframe.groupby("run_id"):
        count = max(
            1,
            math.ceil(len(group) * fraction),
        )

        ranked = group.sort_values(
            ["score", "test_id"],
            ascending=[False, True],
        )

        subset = ranked.head(count)

        selected += len(subset)

        detected += int(
            subset["failed"].sum()
        )

        if group["failed"].sum() > 0:
            failure_runs += 1

            if subset["failed"].sum() > 0:
                detected_runs += 1

    return {
        "fraction": fraction,
        "selected_tests": selected,
        "detected_failures": detected,
        "total_failures": int(labels.sum()),
        "recall": safe_divide(
            detected,
            int(labels.sum()),
        ),
        "precision": safe_divide(
            detected,
            selected,
        ),
        "selection_rate": safe_divide(
            selected,
            len(labels),
        ),
        "failure_run_detection_rate": safe_divide(
            detected_runs,
            failure_runs,
        ),
    }


def choose_threshold(labels, scores):
    labels, scores = validate_scores(
        labels,
        scores,
    )

    if labels.sum() == 0:
        return 0.5

    precision, recall, thresholds = (
        precision_recall_curve(
            labels,
            scores,
        )
    )

    if len(thresholds) == 0:
        return 0.5

    denominator = precision + recall

    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    index = int(
        np.argmax(f1[:-1])
    )

    return float(thresholds[index])


def evaluate_scores(
    labels,
    scores,
    context,
    threshold,
):
    labels, scores = validate_scores(
        labels,
        scores,
    )

    predictions = (
        scores >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    has_both_classes = (
        len(np.unique(labels)) == 2
    )

    metrics = {
        "average_precision": (
            float(
                average_precision_score(
                    labels,
                    scores,
                )
            )
            if labels.sum() > 0
            else None
        ),
        "roc_auc": (
            float(
                roc_auc_score(
                    labels,
                    scores,
                )
            )
            if has_both_classes
            else None
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                labels,
                scores,
            )
        ),
        "mean_score": float(
            scores.mean()
        ),
        "observed_failure_rate": float(
            labels.mean()
        ),
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    for fraction in TOP_FRACTIONS:
        key = f"top_{int(fraction * 100)}"

        metrics[key] = ranking_metrics(
            context,
            scores,
            fraction,
        )

    return metrics


def baseline_scores(
    context,
    training_prevalence,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)

    return {
        "random_ranking": rng.random(
            len(context)
        ),
        "global_failure_probability": np.full(
            len(context),
            training_prevalence,
        ),
        "historical_failure_rate": context[
            "historical_failure_rate"
        ].to_numpy(dtype=np.float64),
        "smoothed_historical_probability": context[
            "smoothed_failure_rate"
        ].to_numpy(dtype=np.float64),
    }


def create_model(name, parameters):
    if name == "logistic_regression":
        return LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_SEED,
            **parameters,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            random_state=RANDOM_SEED,
            n_jobs=MAX_WORKERS,
            **parameters,
        )

    raise ValueError(
        f"Unknown model: {name}"
    )


def tune_model(
    name,
    configurations,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    best_model = None

    best_parameters = None

    best_ap = -1.0

    records = []

    for index, parameters in enumerate(
        configurations,
        start=1,
    ):
        print(
            f"{name}: configuration "
            f"{index}/{len(configurations)}",
            flush=True,
        )

        model = create_model(
            name,
            parameters,
        )

        started = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            time.perf_counter() - started
        )

        scores = model.predict_proba(
            X_validation
        )[:, 1]

        ap = float(
            average_precision_score(
                y_validation,
                scores,
            )
        )

        roc_auc = float(
            roc_auc_score(
                y_validation,
                scores,
            )
        )

        record = {
            "model": name,
            "parameters": parameters,
            "validation_average_precision": ap,
            "validation_roc_auc": roc_auc,
            "training_seconds": training_seconds,
        }

        records.append(record)

        print(
            json.dumps(record),
            flush=True,
        )

        if ap > best_ap:
            best_ap = ap

            best_parameters = parameters

            best_model = model

        del model

        gc.collect()

    if best_model is None:
        raise RuntimeError(
            "Model selection failed"
        )

    return (
        best_model,
        best_parameters,
        records,
    )


def verify_partition_isolation(contexts):
    train = contexts["train"]

    validation = contexts["validation"]

    test = contexts["test"]

    train_builds = set(train["build_key"])

    validation_builds = set(
        validation["build_key"]
    )

    test_builds = set(test["build_key"])

    if train_builds & validation_builds:
        raise ValueError(
            "Train-validation overlap"
        )

    if train_builds & test_builds:
        raise ValueError(
            "Train-test overlap"
        )

    if validation_builds & test_builds:
        raise ValueError(
            "Validation-test overlap"
        )

    train_time = pd.to_datetime(
        train["timestamp"],
        utc=True,
    )

    validation_time = pd.to_datetime(
        validation["timestamp"],
        utc=True,
    )

    test_time = pd.to_datetime(
        test["timestamp"],
        utc=True,
    )

    if train_time.max() >= validation_time.min():
        raise ValueError(
            "Invalid training chronology"
        )

    if validation_time.max() >= test_time.min():
        raise ValueError(
            "Invalid validation chronology"
        )

    return True


def train_project(project):
    started = time.perf_counter()

    result_directory = RESULTS_DIR / project

    model_directory = MODEL_DIR / project

    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train, y_train, train_context = (
        load_partition(project, "train")
    )

    X_validation, y_validation, validation_context = (
        load_partition(project, "validation")
    )

    X_test, y_test, test_context = (
        load_partition(project, "test")
    )

    verify_partition_isolation({
        "train": train_context,
        "validation": validation_context,
        "test": test_context,
    })

    if len({
        X_train.shape[1],
        X_validation.shape[1],
        X_test.shape[1],
    }) != 1:
        raise ValueError(
            "Feature dimension mismatch"
        )

    for labels in [
        y_train,
        y_validation,
        y_test,
    ]:
        if len(np.unique(labels)) != 2:
            raise ValueError(
                "Partition requires both classes"
            )

    training_prevalence = float(
        y_train.mean()
    )

    validation_scores = baseline_scores(
        validation_context,
        training_prevalence,
    )

    tuning_records = []

    trained_models = {}

    selected_parameters = {}

    configurations = {
        "logistic_regression": LOGISTIC_CONFIGS,
        "random_forest": FOREST_CONFIGS,
    }

    for name, grid in configurations.items():
        model, parameters, records = tune_model(
            name,
            grid,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

        trained_models[name] = model

        selected_parameters[name] = parameters

        tuning_records.extend(records)

        validation_scores[name] = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        joblib.dump(
            model,
            model_directory / f"{name}.joblib",
        )

    validation_results = {}

    thresholds = {}

    for name, scores in validation_scores.items():
        threshold = choose_threshold(
            y_validation,
            scores,
        )

        thresholds[name] = threshold

        validation_results[name] = evaluate_scores(
            y_validation,
            scores,
            validation_context,
            threshold,
        )

    selected_strategy = max(
        validation_results,
        key=lambda name: (
            validation_results[name][
                "average_precision"
            ]
        ),
    )

    selected_ml_model = max(
        trained_models,
        key=lambda name: (
            validation_results[name][
                "average_precision"
            ]
        ),
    )

    test_scores = baseline_scores(
        test_context,
        training_prevalence,
        seed=RANDOM_SEED + 1,
    )

    for name, model in trained_models.items():
        test_scores[name] = (
            model.predict_proba(X_test)[:, 1]
        )

    test_results = {}

    summary = []

    for name, scores in test_scores.items():
        metrics = evaluate_scores(
            y_test,
            scores,
            test_context,
            thresholds[name],
        )

        test_results[name] = metrics

        summary.append({
            "project": project,
            "model": name,
            "validation_average_precision": (
                validation_results[name][
                    "average_precision"
                ]
            ),
            "test_average_precision": (
                metrics["average_precision"]
            ),
            "test_roc_auc": metrics["roc_auc"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_brier_score": (
                metrics["brier_score"]
            ),
            "recall_at_10": (
                metrics["top_10"]["recall"]
            ),
            "recall_at_20": (
                metrics["top_20"]["recall"]
            ),
            "recall_at_50": (
                metrics["top_50"]["recall"]
            ),
            "selected_strategy": (
                name == selected_strategy
            ),
            "selected_ml_model": (
                name == selected_ml_model
            ),
        })

    predictions = test_context.copy()

    for name, scores in test_scores.items():
        predictions[f"score_{name}"] = scores

    predictions["failure_score"] = (
        test_scores[selected_strategy]
    )

    predictions["predicted_failure"] = (
        predictions["failure_score"]
        >= thresholds[selected_strategy]
    ).astype(np.int8)

    predictions.to_csv(
        result_directory / "test_predictions.csv",
        index=False,
    )

    validation_predictions = (
        validation_context.copy()
    )

    for name, scores in validation_scores.items():
        validation_predictions[
            f"score_{name}"
        ] = scores

    validation_predictions["failure_score"] = (
        validation_scores[selected_strategy]
    )

    validation_predictions["predicted_failure"] = (
        validation_predictions["failure_score"]
        >= thresholds[selected_strategy]
    ).astype(np.int8)

    validation_predictions.to_csv(
        result_directory / "validation_predictions.csv",
        index=False,
    )

    pd.DataFrame(tuning_records).to_csv(
        result_directory / "hyperparameter_search.csv",
        index=False,
    )

    pd.DataFrame(summary).to_csv(
        result_directory / "model_summary.csv",
        index=False,
    )

    joblib.dump(
        trained_models[selected_ml_model],
        model_directory / "final_model.joblib",
    )

    report = {
        "project": project,
        "training_rows": len(y_train),
        "validation_rows": len(y_validation),
        "test_rows": len(y_test),
        "training_failures": int(y_train.sum()),
        "validation_failures": int(y_validation.sum()),
        "test_failures": int(y_test.sum()),
        "feature_count": int(X_train.shape[1]),
        "training_prevalence": training_prevalence,
        "selection_metric": "average_precision",
        "selection_partition": "validation",
        "selected_strategy": selected_strategy,
        "selected_ml_model": selected_ml_model,
        "selected_parameters": selected_parameters,
        "thresholds": thresholds,
        "model_parameters_frozen": True,
        "probability_calibration": "not_performed",
        "evaluation_mode": "walk_forward",
        "validation_results": validation_results,
        "test_results": test_results,
        "summary": summary,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    (
        result_directory / "training_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print(
        f"\nSelected strategy: {selected_strategy}"
    )

    print(
        f"Selected ML model: {selected_ml_model}"
    )

    print(
        pd.DataFrame(summary).to_string(index=False)
    )

    return report


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = []

    print("=" * 70)

    print("QUANTCI MULTI-REPOSITORY MODEL TRAINING")

    print("=" * 70)

    for project in PROJECTS:
        print(f"\nProcessing {project}")

        reports.append(
            train_project(project)
        )

        gc.collect()

    summary = pd.DataFrame([
        record
        for report in reports
        for record in report["summary"]
    ])

    summary.to_csv(
        RESULTS_DIR / "combined_model_summary.csv",
        index=False,
    )

    report = {
        "projects": PROJECTS,
        "selection_partition": "validation",
        "selection_metric": "average_precision",
        "repositories": reports,
    }

    (
        RESULTS_DIR / "combined_training_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nCombined Model Summary\n")

    print(
        summary[
            [
                "project",
                "model",
                "validation_average_precision",
                "test_average_precision",
                "test_roc_auc",
                "recall_at_10",
                "recall_at_20",
                "selected_strategy",
            ]
        ].to_string(index=False)
    )

    print(
        "\nMULTI-REPOSITORY MODEL TRAINING COMPLETED"
    )


if __name__ == "__main__":
    main()