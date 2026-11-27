import numpy as np
import pandas as pd
import pytest

from scipy.sparse import issparse

from src.features.text_features import (
    normalize_text,
    tokenize,
    jaccard_similarity,
    FeaturePreprocessor,
    NUMERICAL_COLUMNS,
)


def create_sample_dataset():
    dataframe = pd.DataFrame(
        {
            "commit_message": [
                "Fix broker authentication",
                "Fix broker authentication",
                "Update broker authentication",
                "Update broker authentication",
            ],
            "changed_files_text": [
                "src/AuthenticationBroker.java",
                "src/AuthenticationBroker.java",
                "src/AuthenticationService.java",
                "src/AuthenticationService.java",
            ],
            "test_id": [
                "org.example.AuthenticationTest",
                "org.example.BrokerTest",
                "org.example.AuthenticationServiceTest",
                "org.example.AuthenticationBrokerTest",
            ],
        }
    )

    for column in NUMERICAL_COLUMNS:
        dataframe[column] = 1.0

    return dataframe


def test_normalize_text():
    result = normalize_text(
        "AuthenticationBroker.java"
    )

    assert result == "authentication broker java"


def test_tokenization():
    tokens = tokenize(
        "AuthenticationBroker"
    )

    assert "authentication" in tokens
    assert "broker" in tokens


def test_identical_similarity():
    assert jaccard_similarity(
        "broker test",
        "broker test",
    ) == 1.0


def test_unrelated_similarity():
    assert jaccard_similarity(
        "broker",
        "database",
    ) == 0.0


def test_empty_similarity():
    assert jaccard_similarity(
        "",
        "",
    ) == 0.0


def test_sparse_matrix_generation():
    dataframe = create_sample_dataset()

    preprocessor = FeaturePreprocessor()

    matrix = preprocessor.fit_transform(
        dataframe
    )

    assert issparse(matrix)

    assert matrix.shape[0] == len(dataframe)

    assert matrix.shape[1] == len(
        preprocessor.get_feature_names()
    )


def test_finite_matrix_values():
    dataframe = create_sample_dataset()

    preprocessor = FeaturePreprocessor()

    matrix = preprocessor.fit_transform(
        dataframe
    )

    assert np.isfinite(
        matrix.data
    ).all()


def test_transform_before_fit():
    dataframe = create_sample_dataset()

    preprocessor = FeaturePreprocessor()

    with pytest.raises(RuntimeError):
        preprocessor.transform(
            dataframe
        )


def test_unseen_vocabulary():
    dataframe = create_sample_dataset()

    preprocessor = FeaturePreprocessor()

    preprocessor.fit(dataframe)

    original_vocabulary = dict(
        preprocessor.code_vectorizer.vocabulary_
    )

    unseen = dataframe.copy()

    unseen["commit_message"] = (
        "completelyunseenword"
    )

    matrix = preprocessor.transform(
        unseen
    )

    assert matrix.shape[0] == len(unseen)

    assert (
        preprocessor.code_vectorizer.vocabulary_
        == original_vocabulary
    )


def test_target_exclusion():
    dataframe = create_sample_dataset()

    dataframe["failed"] = [0, 1, 0, 1]

    dataframe["duration_seconds"] = [
        10,
        20,
        30,
        40,
    ]

    preprocessor = FeaturePreprocessor()

    first = preprocessor.fit_transform(
        dataframe
    )

    dataframe["failed"] = [1, 0, 1, 0]

    dataframe["duration_seconds"] = [
        100,
        200,
        300,
        400,
    ]

    second = preprocessor.transform(
        dataframe
    )

    assert (first != second).nnz == 0