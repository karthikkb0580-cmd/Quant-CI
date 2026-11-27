import numpy as np
import pandas as pd
import pytest

from scipy.sparse import csr_matrix

from sklearn.ensemble import RandomForestClassifier

from src.ml.final_evaluation import (
    select_final_model,
    predict_model,
    create_summary,
    save_predictions,
)

from src.ml.metrics import evaluate_predictions


def test_model_selection():
    scores = {
        "baseline": 0.03,
        "logistic_regression": 0.04,
        "random_forest": 0.06,
    }

    assert select_final_model(
        scores
    ) == "random_forest"


def test_baseline_can_win():
    scores = {
        "baseline": 0.08,
        "logistic_regression": 0.04,
        "random_forest": 0.06,
    }

    assert select_final_model(
        scores
    ) == "baseline"


def test_empty_candidates():
    with pytest.raises(ValueError):
        select_final_model({})


def test_invalid_candidate_scores():
    with pytest.raises(ValueError):
        select_final_model(
            {"model": np.nan}
        )


@pytest.fixture
def sample_model():
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

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)

    artifact = {
        "model": model,
        "feature_count": 2,
        "threshold": 0.5,
    }

    return artifact, X


def test_prediction_shape(sample_model):
    artifact, X = sample_model

    scores = predict_model(
        artifact,
        X,
    )

    assert scores.shape == (6,)


def test_prediction_range(sample_model):
    artifact, X = sample_model

    scores = predict_model(
        artifact,
        X,
    )

    assert np.all(scores >= 0)
    assert np.all(scores <= 1)


def test_feature_mismatch(sample_model):
    artifact, _ = sample_model

    invalid_matrix = csr_matrix(
        [[1, 2, 3]]
    )

    with pytest.raises(ValueError):
        predict_model(
            artifact,
            invalid_matrix,
        )


def test_summary_generation():
    context = pd.DataFrame(
        {
            "run_id": ["r1", "r1", "r2", "r2"],
            "test_id": ["t1", "t2", "t1", "t2"],
            "failed": [0, 1, 0, 1],
        }
    )

    scores = np.array(
        [0.1, 0.9, 0.2, 0.8]
    )

    results = {
        "random_forest": evaluate_predictions(
            context,
            scores,
        )
    }

    summary = create_summary(
        results,
        {"random_forest": 0.01},
    )

    assert len(summary) == 1

    assert summary.iloc[0][
        "average_precision"
    ] == 1.0


def test_prediction_export(tmp_path):
    context = pd.DataFrame(
        {
            "run_id": ["r1", "r1"],
            "test_id": ["t1", "t2"],
            "failed": [0, 1],
        }
    )

    scores = {
        "random_forest": np.array(
            [0.2, 0.9]
        )
    }

    path = tmp_path / "predictions.csv"

    dataframe = save_predictions(
        context,
        scores,
        "random_forest",
        0.5,
        path,
    )

    assert path.exists()

    assert dataframe[
        "predicted_failure"
    ].tolist() == [0, 1]

    assert dataframe[
        "failure_score"
    ].tolist() == [0.2, 0.9]