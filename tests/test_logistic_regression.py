import numpy as np
import pytest
import joblib

from scipy.sparse import csr_matrix

from src.ml.logistic_regression import (
    create_model,
    predict_scores,
    find_best_threshold,
)


@pytest.fixture
def sample_data():
    X = csr_matrix(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2],
            [3, 3],
        ],
        dtype=np.float32,
    )

    y = np.array(
        [0, 0, 0, 1, 1, 1]
    )

    return X, y


def test_create_unweighted_model():
    model = create_model(
        1.0,
        None,
    )

    assert model.C == 1.0
    assert model.class_weight is None


def test_create_balanced_model():
    model = create_model(
        1.0,
        "balanced",
    )

    assert model.class_weight == "balanced"


def test_sparse_training(sample_data):
    X, y = sample_data

    model = create_model(
        1.0,
        None,
    )

    model.fit(X, y)

    assert hasattr(
        model,
        "coef_",
    )


def test_prediction_shape(sample_data):
    X, y = sample_data

    model = create_model(
        1.0,
        None,
    )

    model.fit(X, y)

    scores = predict_scores(
        model,
        X,
    )

    assert scores.shape == (6,)


def test_probability_range(sample_data):
    X, y = sample_data

    model = create_model(
        1.0,
        "balanced",
    )

    model.fit(X, y)

    scores = predict_scores(
        model,
        X,
    )

    assert np.all(scores >= 0)
    assert np.all(scores <= 1)


def test_threshold_selection():
    labels = np.array(
        [0, 0, 1, 1]
    )

    scores = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    threshold = find_best_threshold(
        labels,
        scores,
    )

    assert 0.2 < threshold <= 0.8


def test_model_serialization(
    sample_data,
    tmp_path,
):
    X, y = sample_data

    model = create_model(
        1.0,
        None,
    )

    model.fit(X, y)

    path = tmp_path / "model.joblib"

    joblib.dump(
        model,
        path,
    )

    loaded = joblib.load(
        path
    )

    first = predict_scores(
        model,
        X,
    )

    second = predict_scores(
        loaded,
        X,
    )

    assert np.allclose(
        first,
        second,
    )


def test_reproducible_training(
    sample_data,
):
    X, y = sample_data

    first = create_model(
        1.0,
        None,
    )

    second = create_model(
        1.0,
        None,
    )

    first.fit(X, y)
    second.fit(X, y)

    assert np.allclose(
        first.coef_,
        second.coef_,
    )