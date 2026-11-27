import pandas as pd
import pytest

from src.evaluation.heldout_evaluation import (
    safe_divide,
    validate_data,
    select_by_order,
    select_baseline,
    evaluate_run,
    summarize_results,
)


@pytest.fixture
def dataframe():
    return pd.DataFrame({
        "run_id": ["R1", "R1", "R1"],
        "test_id": ["A", "B", "C"],
        "failed": [1, 0, 1],
        "duration_seconds": [6.0, 3.0, 8.0],
        "estimated_duration_seconds": [4.0, 3.0, 5.0],
        "failure_score": [0.9, 0.2, 0.8],
        "smoothed_failure_rate": [0.5, 0.1, 0.4],
    })


def test_safe_division():
    assert safe_divide(1, 2) == 0.5


def test_zero_denominator():
    assert safe_divide(1, 0) is None


def test_valid_data(dataframe):
    validate_data(dataframe)


def test_invalid_labels(dataframe):
    dataframe.loc[0, "failed"] = 2

    with pytest.raises(ValueError):
        validate_data(dataframe)


def test_duplicate_tests(dataframe):
    duplicated = pd.concat(
        [dataframe, dataframe]
    )

    with pytest.raises(ValueError):
        validate_data(duplicated)


def test_budget_selection(dataframe):
    selected = select_by_order(
        dataframe,
        7,
    )

    assert selected == ["A", "B"]


def test_zero_budget(dataframe):
    assert select_by_order(
        dataframe,
        0,
    ) == []


def test_shortest_first(dataframe):
    selected = select_baseline(
        dataframe,
        "shortest_first",
        3,
        42,
    )

    assert selected == ["B"]


def test_ml_risk(dataframe):
    selected = select_baseline(
        dataframe,
        "ml_risk",
        9,
        42,
    )

    assert selected == ["A", "C"]


def test_historical_risk(dataframe):
    selected = select_baseline(
        dataframe,
        "historical_risk",
        4,
        42,
    )

    assert selected == ["A"]


def test_random_reproducibility(dataframe):
    first = select_baseline(
        dataframe,
        "random",
        7,
        42,
    )

    second = select_baseline(
        dataframe,
        "random",
        7,
        42,
    )

    assert first == second


def test_failure_recall(dataframe):
    result = evaluate_run(
        dataframe,
        dataframe,
        ["A"],
        4,
    )

    assert result["failure_recall"] == 0.5

    assert result["detected_failures"] == 1


def test_actual_budget_overrun(dataframe):
    result = evaluate_run(
        dataframe,
        dataframe,
        ["A"],
        4,
    )

    assert result["actual_budget_overrun"]

    assert result["actual_overrun_seconds"] == 2


def test_estimated_budget_violation(dataframe):
    with pytest.raises(RuntimeError):
        evaluate_run(
            dataframe,
            dataframe,
            ["A", "B", "C"],
            4,
        )


def test_candidate_recall(dataframe):
    candidates = dataframe.iloc[:2]

    result = evaluate_run(
        dataframe,
        candidates,
        ["A"],
        4,
    )

    assert result["candidate_recall"] == 0.5


def test_unknown_selection(dataframe):
    with pytest.raises(ValueError):
        evaluate_run(
            dataframe,
            dataframe,
            ["Unknown"],
            10,
        )


def test_test_reduction(dataframe):
    result = evaluate_run(
        dataframe,
        dataframe,
        ["A"],
        4,
    )

    assert result["test_reduction"] == pytest.approx(
        2 / 3
    )


def test_runtime_reduction(dataframe):
    result = evaluate_run(
        dataframe,
        dataframe,
        ["A"],
        4,
    )

    assert result["actual_runtime_reduction"] == pytest.approx(
        11 / 17
    )


def test_summary(dataframe):
    result = evaluate_run(
        dataframe,
        dataframe,
        ["A"],
        4,
    )

    result.update({
        "algorithm": "greedy",
        "budget_seconds": 4,
        "solver_seconds": 0.01,
    })

    summary = summarize_results(
        pd.DataFrame([result])
    )

    assert len(summary) == 1

    assert summary.iloc[0]["failure_recall"] == 0.5

    assert summary.iloc[0]["detected_failures"] == 1