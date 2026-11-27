import numpy as np
import pandas as pd
import pytest

from src.ml.metrics import (
    validate_predictions,
    classification_metrics,
    ranking_metrics,
    evaluate_predictions,
)


def sample_context():
    return pd.DataFrame(
        {
            "run_id": [
                "run1",
                "run1",
                "run1",
                "run1",
                "run2",
                "run2",
                "run2",
                "run2",
            ],
            "test_id": [
                "test1",
                "test2",
                "test3",
                "test4",
                "test1",
                "test2",
                "test3",
                "test4",
            ],
            "failed": [
                1, 0, 0, 0,
                0, 1, 0, 0,
            ],
        }
    )


def test_perfect_classification():
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8]

    result = classification_metrics(
        labels,
        scores,
    )

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["average_precision"] == 1.0


def test_majority_prediction():
    labels = [0, 0, 0, 1]
    scores = [0, 0, 0, 0]

    result = classification_metrics(
        labels,
        scores,
    )

    assert result["accuracy"] == 0.75
    assert result["recall"] == 0.0
    assert result["false_negatives"] == 1
    assert result["roc_auc"] == 0.5
    assert result["average_precision"] == 0.25


def test_invalid_length():
    with pytest.raises(ValueError):
        validate_predictions(
            [0, 1],
            [0.5],
        )


def test_invalid_scores():
    with pytest.raises(ValueError):
        validate_predictions(
            [0, 1],
            [0.1, 1.5],
        )


def test_invalid_labels():
    with pytest.raises(ValueError):
        validate_predictions(
            [0, 2],
            [0.1, 0.5],
        )


def test_nonfinite_scores():
    with pytest.raises(ValueError):
        validate_predictions(
            [0, 1],
            [0.1, np.nan],
        )


def test_perfect_ranking():
    context = sample_context()

    scores = [
        0.9, 0.2, 0.1, 0.3,
        0.1, 0.9, 0.2, 0.3,
    ]

    result = ranking_metrics(
        context,
        scores,
        fractions=[0.25],
    )

    assert result[
        "top_25_percent"
    ]["recall"] == 1.0

    assert result[
        "top_25_percent"
    ]["selected_tests"] == 2


def test_full_selection():
    context = sample_context()

    scores = np.zeros(
        len(context)
    )

    result = ranking_metrics(
        context,
        scores,
        fractions=[1.0],
    )

    assert result[
        "top_100_percent"
    ]["recall"] == 1.0

    assert result[
        "top_100_percent"
    ]["selected_tests"] == len(context)


def test_deterministic_ties():
    context = sample_context()

    scores = np.zeros(
        len(context)
    )

    first = ranking_metrics(
        context,
        scores,
    )

    second = ranking_metrics(
        context,
        scores,
    )

    assert first == second


def test_complete_evaluation():
    context = sample_context()

    scores = np.full(
        len(context),
        0.5,
    )

    result = evaluate_predictions(
        context,
        scores,
    )

    assert "classification" in result
    assert "ranking" in result