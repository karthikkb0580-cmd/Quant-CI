import numpy as np
import pandas as pd
import pytest

from scipy import sparse

from src.ml.cross_repository_evaluation import (
    experiment_type,
    calculate_ap_change,
    validate_compatibility,
    predict_in_batches,
    summarize_experiment,
)


class DummyPreprocessor:
    def feature_names(self):
        return ["a", "b"]

    def transform(self, dataframe):
        return sparse.csr_matrix(
            dataframe[["a", "b"]].to_numpy()
        )


class DummyModel:
    n_features_in_ = 2

    classes_ = np.array([0, 1])

    def predict_proba(self, matrix):
        scores = np.full(
            matrix.shape[0],
            0.25,
        )

        return np.column_stack([
            1 - scores,
            scores,
        ])


@pytest.fixture
def dataframe():
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [2, 3, 4, 5, 6],
    })


@pytest.fixture
def metrics():
    return {
        "average_precision": 0.2,
        "roc_auc": 0.8,
        "precision": 0.1,
        "recall": 0.5,
        "f1": 0.1667,
        "brier_score": 0.01,
        "top_10": {
            "recall": 0.5,
            "total_failures": 10,
            "detected_failures": 5,
            "failure_run_detection_rate": 0.6,
        },
        "top_20": {
            "recall": 0.7,
        },
        "top_50": {
            "recall": 0.9,
        },
    }


def test_within_repository():
    assert experiment_type(
        "karaf",
        "karaf",
    ) == "within_repository"


def test_cross_repository():
    assert experiment_type(
        "karaf",
        "hbase",
    ) == "cross_repository"


def test_positive_ap_change():
    assert calculate_ap_change(
        0.2,
        0.1,
    ) == pytest.approx(1.0)


def test_negative_ap_change():
    assert calculate_ap_change(
        0.1,
        0.2,
    ) == pytest.approx(-0.5)


def test_equal_ap():
    assert calculate_ap_change(
        0.1,
        0.1,
    ) == 0.0


def test_zero_reference():
    assert calculate_ap_change(
        0.1,
        0.0,
    ) is None


def test_model_compatibility():
    assert validate_compatibility(
        DummyPreprocessor(),
        DummyModel(),
    )


def test_incompatible_dimensions():
    model = DummyModel()

    model.n_features_in_ = 3

    with pytest.raises(ValueError):
        validate_compatibility(
            DummyPreprocessor(),
            model,
        )


def test_invalid_classes():
    model = DummyModel()

    model.classes_ = np.array([1, 2])

    with pytest.raises(ValueError):
        validate_compatibility(
            DummyPreprocessor(),
            model,
        )


def test_batch_prediction(dataframe):
    scores, elapsed = predict_in_batches(
        dataframe,
        DummyPreprocessor(),
        {"model": DummyModel()},
        batch_size=2,
    )

    assert len(scores["model"]) == 5

    assert elapsed >= 0


def test_batch_scores(dataframe):
    scores, _ = predict_in_batches(
        dataframe,
        DummyPreprocessor(),
        {"model": DummyModel()},
        batch_size=2,
    )

    assert np.allclose(
        scores["model"],
        0.25,
    )


def test_multiple_models(dataframe):
    scores, _ = predict_in_batches(
        dataframe,
        DummyPreprocessor(),
        {
            "first": DummyModel(),
            "second": DummyModel(),
        },
        batch_size=2,
    )

    assert len(scores) == 2

    assert np.array_equal(
        scores["first"],
        scores["second"],
    )


def test_invalid_batch_size(dataframe):
    with pytest.raises(ValueError):
        predict_in_batches(
            dataframe,
            DummyPreprocessor(),
            {"model": DummyModel()},
            batch_size=0,
        )


def test_empty_dataset(dataframe):
    with pytest.raises(ValueError):
        predict_in_batches(
            dataframe.iloc[:0],
            DummyPreprocessor(),
            {"model": DummyModel()},
        )


def test_summary(metrics):
    result = summarize_experiment(
        "karaf",
        "hbase",
        "random_forest",
        metrics,
        {"average_precision": 0.4},
        0.1,
        1.0,
        1000,
    )

    assert result["test_rows"] == 1000

    assert result["relative_ap_change"] == (
        pytest.approx(-0.5)
    )

    assert result["beats_target_historical"]

    assert result["detected_failures_at_10"] == 5