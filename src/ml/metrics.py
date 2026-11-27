import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.config import (
    PREDICTION_THRESHOLD,
    TOP_K_FRACTIONS,
    RANDOM_SEED,
)


def validate_predictions(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("Expected one-dimensional arrays.")

    if len(labels) != len(scores):
        raise ValueError("Label and score length mismatch.")

    if len(labels) == 0:
        raise ValueError("Empty predictions.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid labels.")

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction scores.")

    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Scores must be between 0 and 1.")

    return labels, scores


def classification_metrics(
    labels,
    scores,
    threshold=PREDICTION_THRESHOLD,
):
    labels, scores = validate_predictions(
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

    both_classes = len(
        np.unique(labels)
    ) == 2

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(labels, predictions)
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
        "f1_score": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": (
            float(roc_auc_score(labels, scores))
            if both_classes
            else None
        ),
        "average_precision": (
            float(
                average_precision_score(
                    labels,
                    scores,
                )
            )
            if both_classes
            else None
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_failures": int(
            predictions.sum()
        ),
    }


def ranking_metrics(
    context,
    scores,
    fractions=TOP_K_FRACTIONS,
):
    labels, scores = validate_predictions(
        context["failed"].to_numpy(),
        scores,
    )

    dataframe = context[
        ["run_id", "test_id", "failed"]
    ].copy()

    dataframe["score"] = scores

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    dataframe["tie_breaker"] = rng.random(
        len(dataframe)
    )

    total_failures = int(labels.sum())

    results = {}

    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Invalid selection fraction.")

        selected_tests = 0
        detected_failures = 0
        runs_with_failures = 0
        detected_failure_runs = 0

        for _, run in dataframe.groupby(
            "run_id",
            sort=False,
        ):
            ranked = run.sort_values(
                ["score", "tie_breaker"],
                ascending=[False, True],
            )

            count = max(
                1,
                int(np.ceil(len(ranked) * fraction)),
            )

            selected = ranked.head(count)

            failures = int(
                selected["failed"].sum()
            )

            selected_tests += len(selected)
            detected_failures += failures

            if int(run["failed"].sum()) > 0:
                runs_with_failures += 1

                if failures > 0:
                    detected_failure_runs += 1

        key = f"top_{int(fraction * 100)}_percent"

        results[key] = {
            "selected_tests": selected_tests,
            "detected_failures": detected_failures,
            "total_failures": total_failures,
            "recall": (
                detected_failures / total_failures
                if total_failures
                else 0.0
            ),
            "precision": (
                detected_failures / selected_tests
                if selected_tests
                else 0.0
            ),
            "selection_rate": (
                selected_tests / len(dataframe)
            ),
            "failure_runs_detected": (
                detected_failure_runs
            ),
            "total_failure_runs": (
                runs_with_failures
            ),
            "failure_run_detection_rate": (
                detected_failure_runs
                / runs_with_failures
                if runs_with_failures
                else 0.0
            ),
        }

    return results


def evaluate_predictions(
    context,
    scores,
    threshold=PREDICTION_THRESHOLD,
):
    return {
        "classification": classification_metrics(
            context["failed"].to_numpy(),
            scores,
            threshold,
        ),
        "ranking": ranking_metrics(
            context,
            scores,
        ),
    }