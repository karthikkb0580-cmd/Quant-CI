import re

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.config import (
    MAX_TFIDF_FEATURES,
    MAX_TEST_TFIDF_FEATURES,
    TFIDF_MIN_DF,
)

from src.features.feature_engineering import HISTORICAL_COLUMNS


CODE_CHANGE_COLUMNS = [
    "num_commits",
    "num_changed_files",
    "num_additions",
    "num_deletions",
    "num_changed_lines",
    "num_added_files",
    "num_modified_files",
    "num_removed_files",
]

NUMERICAL_COLUMNS = (
    HISTORICAL_COLUMNS
    + CODE_CHANGE_COLUMNS
)

TEXT_COLUMNS = [
    "commit_message",
    "changed_files_text",
    "test_id",
]

SIMILARITY_COLUMNS = [
    "test_commit_similarity",
    "test_file_similarity",
]


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"([A-Z]+)([A-Z][a-z])",
        r"\1 \2",
        text,
    )

    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text,
    )

    text = re.sub(
        r"[^a-zA-Z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.lower().strip()


def tokenize(text):
    return set(
        normalize_text(text).split()
    )


def jaccard_similarity(first, second):
    first_tokens = tokenize(first)
    second_tokens = tokenize(second)

    union = first_tokens | second_tokens

    if not union:
        return 0.0

    return len(
        first_tokens & second_tokens
    ) / len(union)


class FeaturePreprocessor:

    def __init__(self):
        self.code_vectorizer = TfidfVectorizer(
            max_features=MAX_TFIDF_FEATURES,
            min_df=TFIDF_MIN_DF,
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
        )

        self.test_vectorizer = TfidfVectorizer(
            max_features=MAX_TEST_TFIDF_FEATURES,
            min_df=TFIDF_MIN_DF,
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
        )

        self.scaler = StandardScaler()

        self.fitted = False

    def validate_input(self, dataframe):
        required = set(
            NUMERICAL_COLUMNS + TEXT_COLUMNS
        )

        missing = required - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"Missing features: {sorted(missing)}"
            )

        if dataframe.empty:
            raise ValueError("Empty feature dataset.")

    def prepare_code_text(self, dataframe):
        messages = (
            dataframe["commit_message"]
            .fillna("")
            .astype(str)
        )

        paths = (
            dataframe["changed_files_text"]
            .fillna("")
            .astype(str)
        )

        combined = messages + " " + paths

        return combined.map(normalize_text)

    def prepare_test_text(self, dataframe):
        return (
            dataframe["test_id"]
            .fillna("")
            .astype(str)
            .map(normalize_text)
        )

    def prepare_numerical(self, dataframe):
        numerical = dataframe[
            NUMERICAL_COLUMNS
        ].apply(
            pd.to_numeric,
            errors="raise",
        )

        values = numerical.to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Invalid numerical features."
            )

        return values

    def prepare_similarity(self, dataframe):
        tests = (
            dataframe["test_id"]
            .fillna("")
            .astype(str)
        )

        messages = (
            dataframe["commit_message"]
            .fillna("")
            .astype(str)
        )

        paths = (
            dataframe["changed_files_text"]
            .fillna("")
            .astype(str)
        )

        commit_similarity = [
            jaccard_similarity(test, message)
            for test, message in zip(
                tests,
                messages,
            )
        ]

        file_similarity = [
            jaccard_similarity(test, path)
            for test, path in zip(
                tests,
                paths,
            )
        ]

        return np.column_stack(
            [
                commit_similarity,
                file_similarity,
            ]
        ).astype(np.float32)

    def fit(self, dataframe):
        self.validate_input(dataframe)

        code_documents = dataframe[
            [
                "commit_message",
                "changed_files_text",
            ]
        ].drop_duplicates()

        test_documents = dataframe[
            ["test_id"]
        ].drop_duplicates()

        code_text = self.prepare_code_text(
            code_documents
        )

        test_text = self.prepare_test_text(
            test_documents
        )

        numerical = self.prepare_numerical(
            dataframe
        )

        self.code_vectorizer.fit(code_text)

        self.test_vectorizer.fit(test_text)

        self.scaler.fit(numerical)

        self.fitted = True

        return self

    def transform(self, dataframe):
        if not self.fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted."
            )

        self.validate_input(dataframe)

        numerical = self.prepare_numerical(
            dataframe
        )

        numerical = self.scaler.transform(
            numerical
        ).astype(np.float32)

        code_text = self.prepare_code_text(
            dataframe
        )

        test_text = self.prepare_test_text(
            dataframe
        )

        code_matrix = (
            self.code_vectorizer.transform(
                code_text
            )
        )

        test_matrix = (
            self.test_vectorizer.transform(
                test_text
            )
        )

        similarity = self.prepare_similarity(
            dataframe
        )

        matrix = hstack(
            [
                csr_matrix(numerical),
                code_matrix,
                test_matrix,
                csr_matrix(similarity),
            ],
            format="csr",
            dtype=np.float32,
        )

        matrix.eliminate_zeros()

        if not np.isfinite(matrix.data).all():
            raise ValueError(
                "Invalid sparse matrix values."
            )

        return matrix

    def fit_transform(self, dataframe):
        return self.fit(
            dataframe
        ).transform(dataframe)

    def get_feature_names(self):
        if not self.fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted."
            )

        code_names = [
            f"code_tfidf_{name}"
            for name in
            self.code_vectorizer.get_feature_names_out()
        ]

        test_names = [
            f"test_tfidf_{name}"
            for name in
            self.test_vectorizer.get_feature_names_out()
        ]

        return (
            NUMERICAL_COLUMNS
            + code_names
            + test_names
            + SIMILARITY_COLUMNS
        )