import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.multi_repository_matrices import (
    RepositoryPreprocessor,
    load_partition,
    CONTEXT_COLUMNS,
)

from src.ml.multi_repository_training import (
    evaluate_scores,
    baseline_scores,
)


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    BASE_DIR / "models" / "multi_repository"
)

TRAINING_DIR = (
    BASE_DIR
    / "results"
    / "multi_repository"
    / "training"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "multi_repository"
    / "transfer"
)

PROJECTS = ["karaf", "hbase"]

MODELS = [
    "logistic_regression",
    "random_forest",
]

BATCH_SIZE = 10000


def load_report(project):
    path = (
        TRAINING_DIR
        / project
        / "training_report.json"
    )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_preprocessor(project):
    path = (
        MODEL_DIR
        / project
        / "preprocessor.joblib"
    )

    preprocessor = joblib.load(path)

    if not isinstance(
        preprocessor,
        RepositoryPreprocessor,
    ):
        raise TypeError(
            "Unexpected preprocessor type"
        )

    if not preprocessor.fitted:
        raise ValueError(
            "Preprocessor is not fitted"
        )

    return preprocessor


def load_model(project, model_name):
    path = (
        MODEL_DIR
        / project
        / f"{model_name}.joblib"
    )

    return joblib.load(path)


def validate_compatibility(
    preprocessor,
    model,
):
    feature_count = len(
        preprocessor.feature_names()
    )

    if feature_count != model.n_features_in_:
        raise ValueError(
            "Model and preprocessor mismatch"
        )

    if list(model.classes_) != [0, 1]:
        raise ValueError(
            "Unexpected model classes"
        )

    return True


def predict_in_batches(
    dataframe,
    preprocessor,
    models,
    batch_size=BATCH_SIZE,
):
    if batch_size <= 0:
        raise ValueError(
            "Batch size must be positive"
        )

    if dataframe.empty:
        raise ValueError(
            "Empty evaluation dataset"
        )

    scores = {
        name: []
        for name in models
    }

    for model in models.values():
        validate_compatibility(
            preprocessor,
            model,
        )

    started = time.perf_counter()

    for start in range(
        0,
        len(dataframe),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(dataframe),
        )

        batch = dataframe.iloc[start:end]

        matrix = preprocessor.transform(
            batch
        )

        for name, model in models.items():
            probabilities = model.predict_proba(
                matrix
            )[:, 1]

            scores[name].append(
                probabilities
            )

        del matrix

        print(
            f"Predicted {end:,}"
            f"/{len(dataframe):,}",
            flush=True,
        )

    elapsed = (
        time.perf_counter() - started
    )

    return {
        name: np.concatenate(values)
        for name, values in scores.items()
    }, elapsed


def experiment_type(source, target):
    if source == target:
        return "within_repository"

    return "cross_repository"


def calculate_ap_change(
    transferred_ap,
    reference_ap,
):
    if reference_ap == 0:
        return None

    return (
        transferred_ap - reference_ap
    ) / reference_ap


def summarize_experiment(
    source,
    target,
    model_name,
    metrics,
    reference_metrics,
    historical_ap,
    prediction_seconds,
    rows,
):
    ap = metrics["average_precision"]

    reference_ap = reference_metrics[
        "average_precision"
    ]

    return {
        "source_repository": source,
        "target_repository": target,
        "experiment_type": experiment_type(
            source,
            target,
        ),
        "model": model_name,
        "test_rows": rows,
        "test_failures": metrics[
            "top_10"
        ]["total_failures"],
        "average_precision": ap,
        "roc_auc": metrics["roc_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "brier_score": metrics["brier_score"],
        "recall_at_10": metrics[
            "top_10"
        ]["recall"],
        "recall_at_20": metrics[
            "top_20"
        ]["recall"],
        "recall_at_50": metrics[
            "top_50"
        ]["recall"],
        "detected_failures_at_10": metrics[
            "top_10"
        ]["detected_failures"],
        "failure_run_detection_at_10": metrics[
            "top_10"
        ]["failure_run_detection_rate"],
        "target_native_ap": reference_ap,
        "target_historical_ap": historical_ap,
        "relative_ap_change": calculate_ap_change(
            ap,
            reference_ap,
        ),
        "beats_target_historical": (
            ap > historical_ap
        ),
        "prediction_seconds_shared": (
            prediction_seconds
        ),
    }


def evaluate_pair(
    source,
    target,
    dataframe,
):
    print(
        f"\nSOURCE: {source}"
        f" | TARGET: {target}"
    )

    source_report = load_report(source)

    target_report = load_report(target)

    preprocessor = load_preprocessor(
        source
    )

    models = {
        name: load_model(source, name)
        for name in MODELS
    }

    context = dataframe[
        CONTEXT_COLUMNS
    ].copy()

    labels = context[
        "failed"
    ].to_numpy(dtype=np.int8)

    if not context["project"].eq(target).all():
        raise ValueError(
            "Target repository mismatch"
        )

    scores, elapsed = predict_in_batches(
        dataframe,
        preprocessor,
        models,
    )

    predictions = context.copy()

    baseline = baseline_scores(
        context,
        source_report["training_prevalence"],
    )

    historical_ap = target_report[
        "test_results"
    ][
        "historical_failure_rate"
    ][
        "average_precision"
    ]

    records = []

    details = {}

    for name in MODELS:
        threshold = source_report[
            "thresholds"
        ][name]

        metrics = evaluate_scores(
            labels,
            scores[name],
            context,
            threshold,
        )

        reference = target_report[
            "test_results"
        ][name]

        record = summarize_experiment(
            source,
            target,
            name,
            metrics,
            reference,
            historical_ap,
            elapsed,
            len(labels),
        )

        if source == target:
            if not np.isclose(
                metrics["average_precision"],
                reference["average_precision"],
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError(
                    "Native model reproduction failed"
                )

        records.append(record)

        details[name] = metrics

        predictions[
            f"score_{name}"
        ] = scores[name]

        print(
            json.dumps(record, indent=4)
        )

    for name, values in baseline.items():
        predictions[
            f"score_{name}"
        ] = values

    output = (
        OUTPUT_DIR
        / f"{source}_to_{target}"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output / "transfer_predictions.csv",
        index=False,
    )

    report = {
        "source_repository": source,
        "target_repository": target,
        "experiment_type": experiment_type(
            source,
            target,
        ),
        "preprocessor_repository": source,
        "model_repository": source,
        "target_partition": "test",
        "target_rows": len(labels),
        "target_failures": int(labels.sum()),
        "model_retrained": False,
        "preprocessor_refitted": False,
        "threshold_source": "source_validation",
        "target_history_available": True,
        "evaluation_mode": "walk_forward",
        "results": details,
        "summary": records,
    }

    (
        output / "transfer_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    return records


def main():
    started = time.perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    print("=" * 70)
    print("QUANTCI CROSS-REPOSITORY EVALUATION")
    print("=" * 70)

    for target in PROJECTS:
        print(
            f"\nLoading target repository: {target}"
        )

        dataframe = load_partition(
            target,
            "test",
        )

        for source in PROJECTS:
            records.extend(
                evaluate_pair(
                    source,
                    target,
                    dataframe,
                )
            )

            gc.collect()

        del dataframe

        gc.collect()

    summary = pd.DataFrame(records)

    summary.to_csv(
        OUTPUT_DIR / "transfer_summary.csv",
        index=False,
    )

    report = {
        "projects": PROJECTS,
        "models": MODELS,
        "total_experiments": len(records),
        "model_retraining": False,
        "target_preprocessor_refitting": False,
        "target_history_available": True,
        "evaluation_design": (
            "retrospective_cross_repository_transfer"
        ),
        "results": records,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    (
        OUTPUT_DIR / "transfer_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nTransfer Evaluation Summary\n")

    print(
        summary[
            [
                "source_repository",
                "target_repository",
                "model",
                "average_precision",
                "roc_auc",
                "recall_at_10",
                "recall_at_20",
                "relative_ap_change",
                "beats_target_historical",
            ]
        ].to_string(index=False)
    )

    print(
        "\nCROSS-REPOSITORY EVALUATION COMPLETED"
    )


if __name__ == "__main__":
    main()