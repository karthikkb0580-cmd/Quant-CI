from pathlib import Path

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

DATASET_NAME = "LRTS"
PROJECT_NAME = "activemq"

LRTS_REPOSITORY = RAW_DATA_DIR / "LRTS"

ACTIVEMQ_ROOT = (
    RAW_DATA_DIR
    / "activemq"
    / "long_running_test_suites"
)

TEST_RESULTS_DIR = (
    ACTIVEMQ_ROOT
    / "processed_test_result"
    / PROJECT_NAME
)

COMMIT_DATA_DIR = (
    ACTIVEMQ_ROOT
    / "shadata"
    / PROJECT_NAME
    / "compare_commits"
)

METADATA_FILE = (
    LRTS_REPOSITORY
    / "artifact"
    / "metadata"
    / "dataset.csv"
)

NORMALIZED_DATA_FILE = (
    PROCESSED_DATA_DIR / "activemq_normalized.csv"
)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

MAX_TFIDF_FEATURES = 500

DEFAULT_BUDGET_SECONDS = 120
REDUNDANCY_PENALTY = 0.1

MAX_QAOA_CANDIDATES = 10
QAOA_REPS = 1
QAOA_SHOTS = 1024
QAOA_MAX_ITERATIONS = 50

MAX_WORKERS = 2
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

TRAIN_DATA_FILE = SPLITS_DIR / "train.csv"
VALIDATION_DATA_FILE = SPLITS_DIR / "validation.csv"
TEST_DATA_FILE = SPLITS_DIR / "test.csv"

SPLIT_REPORT_FILE = RESULTS_DIR / "split_report.json"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"

TRAIN_FEATURES_FILE = FEATURES_DIR / "train_features.csv"
VALIDATION_FEATURES_FILE = FEATURES_DIR / "validation_features.csv"
TEST_FEATURES_FILE = FEATURES_DIR / "test_features.csv"

HISTORICAL_FEATURE_REPORT = RESULTS_DIR / "historical_feature_report.json"

DEFAULT_TEST_DURATION = 10.0
MIN_ESTIMATED_DURATION = 0.1

FAILURE_PRIOR_ALPHA = 1.0
FAILURE_PRIOR_BETA = 99.0

UNKNOWN_HISTORY_DAYS = -1.0
MATRICES_DIR = PROCESSED_DATA_DIR / "matrices"

TRAIN_MATRIX_FILE = MATRICES_DIR / "X_train.npz"
VALIDATION_MATRIX_FILE = MATRICES_DIR / "X_validation.npz"
TEST_MATRIX_FILE = MATRICES_DIR / "X_test.npz"

TRAIN_LABELS_FILE = MATRICES_DIR / "y_train.npy"
VALIDATION_LABELS_FILE = MATRICES_DIR / "y_validation.npy"
TEST_LABELS_FILE = MATRICES_DIR / "y_test.npy"

TRAIN_CONTEXT_FILE = MATRICES_DIR / "train_context.csv"
VALIDATION_CONTEXT_FILE = MATRICES_DIR / "validation_context.csv"
TEST_CONTEXT_FILE = MATRICES_DIR / "test_context.csv"

PREPROCESSOR_FILE = MODELS_DIR / "feature_preprocessor.joblib"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.json"

TFIDF_REPORT_FILE = RESULTS_DIR / "tfidf_feature_report.json"

MAX_TEST_TFIDF_FEATURES = 200
TFIDF_MIN_DF = 2

BASELINE_REPORT_FILE = RESULTS_DIR / "baseline_report.json"

BASELINE_PREDICTIONS_DIR = RESULTS_DIR / "baseline_predictions"

BASELINE_PLOT_FILE = PLOTS_DIR / "baseline_comparison.png"

PREDICTION_THRESHOLD = 0.5

TOP_K_FRACTIONS = [0.10, 0.20, 0.50]
LOGISTIC_MODEL_FILE = MODELS_DIR / "logistic_regression.joblib"

LOGISTIC_REPORT_FILE = RESULTS_DIR / "logistic_regression_report.json"

LOGISTIC_SUMMARY_FILE = RESULTS_DIR / "logistic_regression_summary.csv"

LOGISTIC_PREDICTIONS_FILE = RESULTS_DIR / "logistic_validation_predictions.csv"

LOGISTIC_COEFFICIENTS_FILE = RESULTS_DIR / "logistic_coefficients.csv"

LOGISTIC_PR_PLOT_FILE = PLOTS_DIR / "logistic_precision_recall.png"

LOGISTIC_ROC_PLOT_FILE = PLOTS_DIR / "logistic_roc_curve.png"

LOGISTIC_COMPARISON_PLOT_FILE = PLOTS_DIR / "logistic_comparison.png"

LOGISTIC_C_VALUES = [0.01, 0.1, 1.0, 10.0]

LOGISTIC_MAX_ITERATIONS = 1000

LOGISTIC_TOLERANCE = 0.0001
RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "random_forest.joblib"

RANDOM_FOREST_REPORT_FILE = RESULTS_DIR / "random_forest_report.json"

RANDOM_FOREST_SUMMARY_FILE = RESULTS_DIR / "random_forest_summary.csv"

RANDOM_FOREST_PREDICTIONS_FILE = RESULTS_DIR / "random_forest_validation_predictions.csv"

RANDOM_FOREST_IMPORTANCE_FILE = RESULTS_DIR / "random_forest_feature_importance.csv"

RANDOM_FOREST_PR_PLOT_FILE = PLOTS_DIR / "random_forest_precision_recall.png"

RANDOM_FOREST_ROC_PLOT_FILE = PLOTS_DIR / "random_forest_roc_curve.png"

RANDOM_FOREST_IMPORTANCE_PLOT_FILE = PLOTS_DIR / "random_forest_feature_importance.png"

RANDOM_FOREST_COMPARISON_PLOT_FILE = PLOTS_DIR / "random_forest_comparison.png"

RANDOM_FOREST_CONFIGURATIONS = [
    (100, 8, 5, None),
    (100, 12, 5, None),
    (150, 16, 5, None),
    (100, 8, 5, "balanced"),
    (100, 12, 5, "balanced"),
    (150, 16, 5, "balanced"),
]
FINAL_MODEL_FILE = MODELS_DIR / "final_model.joblib"

FINAL_MODEL_METADATA_FILE = MODELS_DIR / "final_model_metadata.json"

FINAL_EVALUATION_REPORT = RESULTS_DIR / "final_evaluation_report.json"

FINAL_EVALUATION_SUMMARY = RESULTS_DIR / "final_evaluation_summary.csv"

FINAL_TEST_PREDICTIONS = RESULTS_DIR / "final_test_predictions.csv"

FINAL_VALIDATION_PREDICTIONS = RESULTS_DIR / "final_validation_predictions.csv"

FINAL_PR_PLOT = PLOTS_DIR / "final_precision_recall.png"

FINAL_ROC_PLOT = PLOTS_DIR / "final_roc_curve.png"

FINAL_RECALL_PLOT = PLOTS_DIR / "final_recall_comparison.png"

FINAL_AP_PLOT = PLOTS_DIR / "final_average_precision.png"

FINAL_CONFUSION_PLOT = PLOTS_DIR / "final_confusion_matrix.png"

FINAL_CALIBRATION_PLOT = PLOTS_DIR / "final_calibration.png"

OPTIMIZATION_RESULTS_DIR = RESULTS_DIR / "optimization"

OPTIMIZATION_REPORT_FILE = OPTIMIZATION_RESULTS_DIR / "objective_report.json"

OPTIMIZATION_CANDIDATES_FILE = OPTIMIZATION_RESULTS_DIR / "validation_candidates.csv"

OPTIMIZATION_SIMILARITY_THRESHOLD = 0.3

BENCHMARK_BUDGETS_SECONDS = [60, 120, 300, 600]

MAX_CLASSICAL_CANDIDATES = 200

OPTIMIZATION_TOLERANCE = 1e-8
GREEDY_REPORT_FILE = OPTIMIZATION_RESULTS_DIR / "greedy_report.json"

GREEDY_SELECTION_FILE = OPTIMIZATION_RESULTS_DIR / "greedy_selected_tests.csv"

GREEDY_SUMMARY_FILE = OPTIMIZATION_RESULTS_DIR / "greedy_summary.csv"

GREEDY_TRACE_FILE = OPTIMIZATION_RESULTS_DIR / "greedy_trace.csv"

GREEDY_BUDGET_PLOT = PLOTS_DIR / "greedy_budget_comparison.png"

GREEDY_OBJECTIVE_PLOT = PLOTS_DIR / "greedy_objective_progress.png"

GREEDY_RUNTIME_PLOT = PLOTS_DIR / "greedy_runtime_comparison.png"