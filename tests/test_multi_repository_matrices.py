import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.features.multi_repository_matrices import (
    RepositoryPreprocessor,
    NUMERICAL_COLUMNS,
    normalize_text,
    tokenize_identifier,
    token_similarity,
    validate_features,
    similarity_features,
)


@pytest.fixture
def sample_data():
    records = []

    for index in range(5):
        record = {
            "project": "sample",
            "run_id": f"run{index}",
            "build_key": f"build{index}",
            "stage_id": "single",
            "timestamp": "2023-01-01",
            "test_id": f"DatabaseConnectionTest{index}",
            "failed": index % 2,
            "duration_seconds": 10.0,
            "commit_message": "Fix database connection",
            "changed_files_text": "DatabaseConnection.java",
        }

        record.update({
            column: float(index + 1)
            for column in NUMERICAL_COLUMNS
        })

        records.append(record)

    return pd.DataFrame(records)


def test_identifier_tokenization():
    assert tokenize_identifier(
        "DatabaseConnectionTest"
    ) == [
        "database",
        "connection",
        "test",
    ]


def test_normalize_text():
    assert normalize_text(
        "DatabaseConnection.java"
    ) == "database connection java"


def test_missing_text():
    assert normalize_text(None) == ""


def test_identical_similarity():
    assert token_similarity(
        "database connection",
        "database connection",
    ) == 1.0


def test_disjoint_similarity():
    assert token_similarity(
        "database",
        "network",
    ) == 0.0


def test_empty_similarity():
    assert token_similarity("", "") == 0.0


def test_partial_similarity():
    assert token_similarity(
        "database connection",
        "database test",
    ) == pytest.approx(1 / 3)


def test_valid_features(sample_data):
    result = validate_features(sample_data)

    assert len(result) == 5


def test_missing_columns(sample_data):
    with pytest.raises(ValueError):
        validate_features(
            sample_data.drop(
                columns=["commit_message"]
            )
        )


def test_invalid_labels(sample_data):
    sample_data.loc[0, "failed"] = 2

    with pytest.raises(ValueError):
        validate_features(sample_data)


def test_duplicate_execution(sample_data):
    dataframe = pd.concat([
        sample_data,
        sample_data.iloc[[0]],
    ])

    with pytest.raises(ValueError):
        validate_features(dataframe)


def test_invalid_numerical_features(sample_data):
    sample_data.loc[
        0,
        NUMERICAL_COLUMNS[0],
    ] = np.inf

    with pytest.raises(ValueError):
        validate_features(sample_data)


def test_transform_before_fit(sample_data):
    preprocessor = RepositoryPreprocessor()

    with pytest.raises(ValueError):
        preprocessor.transform(sample_data)


def test_sparse_matrix(sample_data):
    preprocessor = RepositoryPreprocessor()

    matrix = preprocessor.fit_transform(
        sample_data
    )

    assert sparse.isspmatrix_csr(matrix)


def test_matrix_dtype(sample_data):
    preprocessor = RepositoryPreprocessor()

    matrix = preprocessor.fit_transform(
        sample_data
    )

    assert matrix.dtype == np.float32


def test_matrix_dimensions(sample_data):
    preprocessor = RepositoryPreprocessor()

    matrix = preprocessor.fit_transform(
        sample_data
    )

    assert matrix.shape[0] == len(sample_data)

    assert matrix.shape[1] == len(
        preprocessor.feature_names()
    )


def test_finite_matrix(sample_data):
    preprocessor = RepositoryPreprocessor()

    matrix = preprocessor.fit_transform(
        sample_data
    )

    assert np.isfinite(matrix.data).all()


def test_vocabulary_isolation(sample_data):
    preprocessor = RepositoryPreprocessor()

    preprocessor.fit(sample_data)

    original = dict(
        preprocessor.code_vectorizer.vocabulary_
    )

    validation = sample_data.copy()

    validation["commit_message"] = (
        "unseenvocabularyword"
    )

    preprocessor.transform(validation)

    assert original == (
        preprocessor.code_vectorizer.vocabulary_
    )

    assert "unseenvocabularyword" not in original


def test_similarity_dimensions():
    matrix = similarity_features(
        ["database"],
        ["database connection"],
        ["database test"],
    )

    assert matrix.shape == (1, 2)


def test_label_independence(sample_data):
    preprocessor = RepositoryPreprocessor()

    original = preprocessor.fit_transform(
        sample_data
    )

    modified = sample_data.copy()

    modified["failed"] = (
        1 - modified["failed"]
    )

    transformed = preprocessor.transform(
        modified
    )

    assert (original != transformed).nnz == 0