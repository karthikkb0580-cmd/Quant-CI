import numpy as np
import pandas as pd
import pytest

from scipy import sparse

from src.ml.multi_repository_training import (
    safe_divide,
    validate_partition,
    validate_scores,
    ranking_metrics,
    choose_threshold,
    evaluate_scores,
    baseline_scores,
    create_model,
    tune_model,
    verify_partition_isolation,
)


@pytest.fixture
def context():
    return pd.DataFrame({
        "run_id": ["r1", "r1", "r2", "r2"],
        "test_id": ["A", "B", "A", "B"],
        "build_key": ["b1", "b1", "b2", "b2"],
        "timestamp": [
            "2023-01-01",
            "2023-01-01",
            "2023-01-02",
            "2023-01-02",
        ],
        "failed": [1, 0, 0, 1],
        "historical_failure_rate": [
            0.2, 0.0, 0.1, 0.3
        ],
        "smoothed_failure_rate": [
            0.1, 0.01, 0.05, 0.15
        ],
    })


@pytest.fixture
def scores():
    return np.array([
        0.9, 0.1, 0.2, 0.8
    ])


def test_safe_divide():
    assert safe_divide(4, 2) == 2


def test_zero_division():
    assert safe_divide(4, 0) == 0


def test_valid_scores(context, scores):
    labels, result = validate_scores(
        context["failed"],
        scores,
    )

    assert len(labels) == len(result)


def test_invalid_score_length(context):
    with pytest.raises(ValueError):
        validate_scores(
            context["failed"],
            [0.1],
        )


def test_invalid_score_values(context):
    with pytest.raises(ValueError):
        validate_scores(
            context["failed"],
            [0.1, 0.2, 1.5, 0.4],
        )


def test_invalid_score_nan(context):
    with pytest.raises(ValueError):
        validate_scores(
            context["failed"],
            [0.1, np.nan, 0.3, 0.4],
        )


def test_invalid_labels():
    with pytest.raises(ValueError):
        validate_scores(
            [0, 2],
            [0.1, 0.2],
        )


def test_ranking_recall(context, scores):
    result = ranking_metrics(
        context,
        scores,
        0.5,
    )

    assert result["recall"] == 1.0


def test_ranking_selection(context, scores):
    result = ranking_metrics(
        context,
        scores,
        0.5,
    )

    assert result["selected_tests"] == 2


def test_ranking_precision(context, scores):
    result = ranking_metrics(
        context,
        scores,
        0.5,
    )

    assert result["precision"] == 1.0


def test_invalid_fraction(context, scores):
    with pytest.raises(ValueError):
        ranking_metrics(
            context,
            scores,
            1.5,
        )


def test_threshold(context, scores):
    threshold = choose_threshold(
        context["failed"],
        scores,
    )

    assert 0 <= threshold <= 1


def test_perfect_average_precision(context, scores):
    result = evaluate_scores(
        context["failed"],
        scores,
        context,
        0.5,
    )

    assert result["average_precision"] == 1.0


def test_perfect_roc_auc(context, scores):
    result = evaluate_scores(
        context["failed"],
        scores,
        context,
        0.5,
    )

    assert result["roc_auc"] == 1.0


def test_baseline_count(context):
    result = baseline_scores(
        context,
        0.1,
    )

    assert len(result) == 4


def test_random_reproducibility(context):
    first = baseline_scores(
        context,
        0.1,
    )

    second = baseline_scores(
        context,
        0.1,
    )

    assert np.array_equal(
        first["random_ranking"],
        second["random_ranking"],
    )


def test_global_prevalence(context):
    result = baseline_scores(
        context,
        0.1,
    )

    assert np.allclose(
        result["global_failure_probability"],
        0.1,
    )


def test_logistic_creation():
    model = create_model(
        "logistic_regression",
        {
            "C": 1.0,
            "class_weight": "balanced",
        },
    )

    assert model.C == 1.0


def test_forest_creation():
    model = create_model(
        "random_forest",
        {
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "class_weight": None,
        },
    )

    assert model.n_estimators == 10


def test_unknown_model():
    with pytest.raises(ValueError):
        create_model(
            "unknown",
            {},
        )


def test_partition_validation(context):
    matrix = sparse.csr_matrix(
        np.ones((4, 3))
    )

    labels = context[
        "failed"
    ].to_numpy()

    assert validate_partition(
        matrix,
        labels,
        context,
    )


def test_partition_mismatch(context):
    matrix = sparse.csr_matrix(
        np.ones((3, 3))
    )

    with pytest.raises(ValueError):
        validate_partition(
            matrix,
            context["failed"].to_numpy(),
            context,
        )


def test_partition_label_mismatch(context):
    matrix = sparse.csr_matrix(
        np.ones((4, 3))
    )

    with pytest.raises(ValueError):
        validate_partition(
            matrix,
            np.array([0, 1, 1, 0]),
            context,
        )


def test_small_model_training():
    X = sparse.csr_matrix([
        [0.0, 0.0],
        [0.1, 0.2],
        [1.0, 1.0],
        [0.9, 0.8],
    ])

    y = np.array([0, 0, 1, 1])

    model, parameters, records = tune_model(
        "logistic_regression",
        [
            {
                "C": 1.0,
                "class_weight": None,
            }
        ],
        X,
        y,
        X,
        y,
    )

    assert model is not None

    assert len(records) == 1

    assert parameters["C"] == 1.0


def test_partition_overlap(context):
    with pytest.raises(ValueError):
        verify_partition_isolation({
            "train": context,
            "validation": context,
            "test": context,
        })