import joblib
import numpy as np
import pytest

from scipy.sparse import csr_matrix

from src.ml.random_forest import create_model
from src.ml.logistic_regression import predict_scores


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


def test_model_creation():
    model = create_model(
        10,
        5,
        1,
        None,
    )

    assert model.n_estimators == 10
    assert model.max_depth == 5


def test_balanced_model():
    model = create_model(
        10,
        5,
        1,
        "balanced",
    )

    assert model.class_weight == "balanced"


def test_sparse_training(sample_data):
    X, y = sample_data

    model = create_model(
        10,
        5,
        1,
        None,
    )

    model.fit(X, y)

    assert len(model.estimators_) == 10


def test_prediction_shape(sample_data):
    X, y = sample_data

    model = create_model(
        10,
        5,
        1,
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
        10,
        5,
        1,
        "balanced",
    )

    model.fit(X, y)

    scores = predict_scores(
        model,
        X,
    )

    assert np.all(scores >= 0)
    assert np.all(scores <= 1)


def test_feature_importance(sample_data):
    X, y = sample_data

    model = create_model(
        10,
        5,
        1,
        None,
    )

    model.fit(X, y)

    assert len(model.feature_importances_) == 2

    assert np.isclose(
        model.feature_importances_.sum(),
        1.0,
    )


def test_serialization(sample_data, tmp_path):
    X, y = sample_data

    model = create_model(
        10,
        5,
        1,
        None,
    )

    model.fit(X, y)

    path = tmp_path / "random_forest.joblib"

    joblib.dump(
        model,
        path,
    )

    loaded = joblib.load(path)

    assert np.allclose(
        predict_scores(model, X),
        predict_scores(loaded, X),
    )


def test_reproducibility(sample_data):
    X, y = sample_data

    first = create_model(
        10,
        5,
        1,
        None,
    )

    second = create_model(
        10,
        5,
        1,
        None,
    )

    first.fit(X, y)
    second.fit(X, y)

    assert np.allclose(
        predict_scores(first, X),
        predict_scores(second, X),
    )