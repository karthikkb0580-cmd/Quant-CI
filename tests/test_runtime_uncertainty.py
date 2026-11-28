import numpy as np
import pandas as pd
import pytest

from src.optimization.runtime_uncertainty import (
    calculate_duration_ratios,
    conservative_duration,
    evaluate_coverage,
    fit_uncertainty_model,
    safe_divide,
    validate_data,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "run_id": ["run1"] * 5,
        "test_id": ["A", "B", "C", "D", "E"],
        "timestamp": ["2023-01-01"] * 5,
        "failed": [0, 1, 0, 0, 1],
        "failure_score": [
            0.1, 0.8, 0.2, 0.3, 0.9
        ],
        "duration_seconds": [
            10, 15, 20, 25, 30
        ],
        "estimated_duration_seconds": [
            10, 10, 10, 10, 10
        ],
    })


def test_safe_divide():
    assert safe_divide(10, 2) == 5


def test_zero_division():
    assert safe_divide(10, 0) == 0


def test_valid_data(sample_data):
    assert len(validate_data(sample_data)) == 5


def test_missing_column(sample_data):
    with pytest.raises(ValueError):
        validate_data(
            sample_data.drop(
                columns=["failed"]
            )
        )


def test_invalid_failure_label(sample_data):
    sample_data.loc[0, "failed"] = 2

    with pytest.raises(ValueError):
        validate_data(sample_data)


def test_negative_duration(sample_data):
    sample_data.loc[0, "duration_seconds"] = -1

    with pytest.raises(ValueError):
        validate_data(sample_data)


def test_duplicate_execution(sample_data):
    duplicate = pd.concat([
        sample_data,
        sample_data.iloc[[0]],
    ])

    with pytest.raises(ValueError):
        validate_data(duplicate)


def test_duration_ratios():
    result = calculate_duration_ratios(
        [10, 20, 30],
        [10, 10, 10],
    )

    assert np.allclose(
        result,
        [1, 2, 3],
    )


def test_zero_estimated_duration():
    result = calculate_duration_ratios(
        [10],
        [0],
    )

    assert np.isfinite(result).all()


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        calculate_duration_ratios(
            [10, 20],
            [10],
        )


def test_empty_inputs():
    with pytest.raises(ValueError):
        calculate_duration_ratios(
            [],
            [],
        )


def test_invalid_ratio_input():
    with pytest.raises(ValueError):
        calculate_duration_ratios(
            [np.nan],
            [10],
        )


def test_conservative_duration():
    result = conservative_duration(
        [10, 20, 30],
        1.5,
    )

    assert np.allclose(
        result,
        [15, 30, 45],
    )


def test_invalid_factor():
    with pytest.raises(ValueError):
        conservative_duration(
            [10, 20],
            0.5,
        )


def test_quantile_order(sample_data):
    factors = fit_uncertainty_model(
        sample_data
    )

    assert (
        factors["point_estimate"]
        <= factors["p80"]
        <= factors["p90"]
        <= factors["p95"]
    )


def test_factor_lower_bound(sample_data):
    factors = fit_uncertainty_model(
        sample_data
    )

    assert all(
        factor >= 1
        for factor in factors.values()
    )


def test_coverage_bounds(sample_data):
    factors = fit_uncertainty_model(
        sample_data
    )

    coverage = evaluate_coverage(
        sample_data,
        factors,
    )

    assert coverage[
        "individual_duration_coverage"
    ].between(0, 1).all()


def test_coverage_monotonicity(sample_data):
    factors = fit_uncertainty_model(
        sample_data
    )

    coverage = evaluate_coverage(
        sample_data,
        factors,
    )

    values = coverage[
        "individual_duration_coverage"
    ].to_numpy()

    assert np.all(
        np.diff(values) >= 0
    )


def test_point_estimate_factor(sample_data):
    factors = fit_uncertainty_model(
        sample_data
    )

    assert factors["point_estimate"] == 1.0


def test_missing_values(sample_data):
    sample_data.loc[0, "failure_score"] = np.nan

    with pytest.raises(ValueError):
        validate_data(sample_data)