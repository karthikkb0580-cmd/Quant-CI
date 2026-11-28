import numpy as np
import pytest

from src.ml.calibration import (
    evaluate_probabilities,
    expected_calibration_error,
    fit_calibrator,
    predict_calibrated,
    validate_inputs,
)


@pytest.fixture
def sample_data():
    scores = np.array([
        0.01,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.70,
        0.80,
        0.95,
    ])

    labels = np.array([
        0, 0, 0, 0, 0,
        0, 1, 0, 1, 1,
    ])

    return scores, labels


def test_valid_inputs(sample_data):
    scores, labels = sample_data

    x, y = validate_inputs(scores, labels)

    assert len(x) == len(y)


def test_invalid_scores():
    with pytest.raises(ValueError):
        validate_inputs(
            [0.2, 1.5],
            [0, 1],
        )


def test_invalid_labels():
    with pytest.raises(ValueError):
        validate_inputs(
            [0.2, 0.5],
            [0, 2],
        )


def test_empty_inputs():
    with pytest.raises(ValueError):
        validate_inputs([], [])


def test_mismatched_inputs():
    with pytest.raises(ValueError):
        validate_inputs(
            [0.2, 0.5],
            [0],
        )


def test_nan_scores():
    with pytest.raises(ValueError):
        validate_inputs(
            [0.2, np.nan],
            [0, 1],
        )


@pytest.mark.parametrize(
    "method",
    ["uncalibrated", "platt", "isotonic"],
)
def test_probability_bounds(method, sample_data):
    scores, labels = sample_data

    model = fit_calibrator(
        method,
        scores,
        labels,
    )

    probabilities = predict_calibrated(
        method,
        model,
        scores,
    )

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert len(probabilities) == len(scores)


def test_platt_calibration(sample_data):
    scores, labels = sample_data

    model = fit_calibrator(
        "platt",
        scores,
        labels,
    )

    assert model is not None


def test_isotonic_monotonicity(sample_data):
    scores, labels = sample_data

    model = fit_calibrator(
        "isotonic",
        scores,
        labels,
    )

    probabilities = predict_calibrated(
        "isotonic",
        model,
        scores,
    )

    assert np.all(
        np.diff(probabilities) >= 0
    )


def test_calibration_metrics(sample_data):
    scores, labels = sample_data

    metrics = evaluate_probabilities(
        labels,
        scores,
    )

    assert metrics["brier_score"] >= 0
    assert metrics["log_loss"] >= 0
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["average_precision"] <= 1


def test_perfect_calibration_error():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0, 0, 1, 1])

    error = expected_calibration_error(
        labels,
        probabilities,
    )

    assert error == pytest.approx(0.0)


def test_unknown_method(sample_data):
    scores, labels = sample_data

    with pytest.raises(ValueError):
        fit_calibrator(
            "unknown",
            scores,
            labels,
        )


def test_single_class_calibration():
    with pytest.raises(ValueError):
        fit_calibrator(
            "platt",
            [0.1, 0.2, 0.3],
            [0, 0, 0],
        )