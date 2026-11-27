import streamlit as st
import pandas as pd
import plotly.express as px

from src.demo.components.ui import (
    load_theme,
    brand,
    hero,
    section_label,
    metric_card,
    feature_card,
    page_heading,
)

from src.demo.components.data_loader import (
    read_json,
    read_csv,
    show_plot,
    format_number,
    format_percent,
)


st.set_page_config(
    page_title="QuantCI | Intelligence Laboratory",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_theme()


PALETTE = [
    "#F5AD66",
    "#C87842",
    "#FFE0B5",
    "#A39B91",
    "#796047",
    "#D9D9D9",
    "#E7C69C",
]


def style_chart(figure):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#BDBDBD",
            family="Arial",
        ),
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    figure.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)"
    )

    figure.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)"
    )

    return figure


def overview():
    hero()

    evaluation = read_json(
        "final_evaluation_report.json"
    )

    benchmark = read_json(
        "benchmark/unified_benchmark_report.json"
    )

    normalized = read_csv(
        "final_test_predictions.csv"
    )

    section_label("System intelligence")

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "Test executions",
            format_number(
                evaluation.get("test_rows")
            ),
            "Chronological evaluation partition",
        )

    with columns[1]:
        metric_card(
            "ML features",
            format_number(
                evaluation.get("features")
            ),
            "Historical and textual intelligence",
        )

    with columns[2]:
        metric_card(
            "Quantum variables",
            format_number(
                benchmark.get("qubo_variables")
            ),
            "Reduced QUBO representation",
        )

    with columns[3]:
        metric_card(
            "Optimization variants",
            format_number(
                benchmark.get("algorithms_compared")
            ),
            "Unified experimental benchmark",
        )

    section_label("Platform capabilities")

    first, second = st.columns(2)

    with first:
        feature_card(
            "01",
            "Predict before execution",
            "Analyze historical test behavior, commit messages "
            "and source-code changes to estimate test failure risk.",
        )

        feature_card(
            "03",
            "Classical optimization",
            "Select high-value tests under execution constraints "
            "using greedy search and CP-SAT optimization.",
        )

    with second:
        feature_card(
            "02",
            "Intelligent test selection",
            "Prioritize relevant test classes using predicted "
            "risk, estimated duration and redundancy penalties.",
        )

        feature_card(
            "04",
            "Quantum experimentation",
            "Explore QUBO formulations, Ising Hamiltonians, "
            "QAOA and classical slack repair.",
        )

    if not normalized.empty:
        section_label("Available evaluation data")

        st.dataframe(
            normalized.head(10),
            hide_index=True,
            width="stretch",
        )


def dataset_intelligence():
    page_heading(
        "Dataset Intelligence",
        "Explore the real-world Apache ActiveMQ test dataset.",
    )

    predictions = read_csv(
        "final_test_predictions.csv"
    )

    if predictions.empty:
        st.warning(
            "Test prediction artifacts are unavailable."
        )
        return

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "Executions",
            f"{len(predictions):,}",
            "Test partition",
        )

    with columns[1]:
        metric_card(
            "Failed executions",
            int(predictions["failed"].sum()),
            "Observed historical outcomes",
        )

    with columns[2]:
        metric_card(
            "CI runs",
            predictions["run_id"].nunique(),
            "Chronological suite runs",
        )

    with columns[3]:
        metric_card(
            "Unique tests",
            predictions["test_id"].nunique(),
            "Observed test classes",
        )

    section_label("Failure distribution")

    counts = (
        predictions["failed"]
        .value_counts()
        .rename_axis("outcome")
        .reset_index(name="executions")
    )

    counts["outcome"] = counts["outcome"].map(
        {0: "Passed", 1: "Failed"}
    )

    figure = px.bar(
        counts,
        x="outcome",
        y="executions",
        color="outcome",
        log_y=True,
        color_discrete_sequence=PALETTE,
    )

    st.plotly_chart(
        style_chart(figure),
        width="stretch",
    )

    left, right = st.columns(2)

    with left:
        show_plot("duration_distribution.png")

    with right:
        show_plot("monthly_failure_rate.png")

    section_label("Dataset explorer")

    search = st.text_input(
        "Search test classes"
    )

    filtered = predictions

    if search:
        filtered = filtered[
            filtered["test_id"].str.contains(
                search,
                case=False,
                regex=False,
                na=False,
            )
        ]

    failed_only = st.checkbox(
        "Show failed executions only"
    )

    if failed_only:
        filtered = filtered[
            filtered["failed"] == 1
        ]

    st.dataframe(
        filtered.head(1000),
        width="stretch",
        hide_index=True,
    )


def ml_intelligence():
    page_heading(
        "ML Intelligence",
        "Failure-risk prediction and model evaluation.",
    )

    report = read_json(
        "final_evaluation_report.json"
    )

    if not report:
        st.warning(
            "Model evaluation report is unavailable."
        )
        return

    selected = report.get(
        "selected_model_test_results",
        {},
    )

    classification = selected.get(
        "classification",
        {},
    )

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "Selected model",
            str(
                report.get(
                    "selected_model",
                    "Unavailable",
                )
            ).replace("_", " ").title(),
            "Selected using validation data",
        )

    with columns[1]:
        metric_card(
            "ROC AUC",
            format_percent(
                classification.get("roc_auc")
            ),
            "Test partition discrimination",
        )

    with columns[2]:
        metric_card(
            "Average precision",
            format_percent(
                classification.get(
                    "average_precision"
                )
            ),
            "Imbalanced classification metric",
        )

    with columns[3]:
        metric_card(
            "Feature dimensions",
            report.get("features", "—"),
            "Sparse numerical and text features",
        )

    section_label("Model performance")

    results = report.get(
        "test_results",
        {},
    )

    records = []

    if isinstance(results, dict):
        for name, result in results.items():
            if not isinstance(result, dict):
                continue

            metrics = result.get(
                "classification",
                {},
            )

            if "average_precision" in metrics:
                records.append({
                    "model": name,
                    "average_precision": metrics[
                        "average_precision"
                    ],
                    "roc_auc": metrics.get("roc_auc"),
                })

    if records:
        dataframe = pd.DataFrame(records)

        figure = px.bar(
            dataframe,
            x="model",
            y="average_precision",
            color="model",
            color_discrete_sequence=PALETTE,
        )

        st.plotly_chart(
            style_chart(figure),
            width="stretch",
        )

        st.dataframe(
            dataframe,
            hide_index=True,
            width="stretch",
        )

    section_label("Model evaluation report")

    st.json(
        selected,
        expanded=False,
    )

    st.info(
        "Model outputs are uncalibrated risk scores, "
        "not guaranteed failure probabilities."
    )


def test_optimizer():
    page_heading(
        "Test Optimizer",
        "Interactive budget-constrained test selection.",
    )

    from src.optimization.candidates import (
        prepare_candidates,
    )

    from src.optimization.objective import (
        create_problem,
    )

    from src.optimization.greedy import (
        solve_greedy,
    )

    from src.optimization.ortools_solver import (
        solve_ortools,
    )

    predictions = read_csv(
        "final_test_predictions.csv"
    )

    if predictions.empty:
        st.warning(
            "Prediction data is unavailable."
        )
        return

    first, second = st.columns(2)

    with first:
        run_id = st.selectbox(
            "Select CI run",
            predictions["run_id"].unique(),
        )

        budget = st.slider(
            "Execution budget (seconds)",
            min_value=30,
            max_value=600,
            value=120,
            step=30,
        )

    with second:
        algorithm = st.selectbox(
            "Optimization algorithm",
            ["Greedy", "OR-Tools CP-SAT"],
        )

        maximum = st.slider(
            "Maximum candidate tests",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
        )

    penalty = st.slider(
        "Redundancy penalty",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
    )

    run = predictions[
        predictions["run_id"] == run_id
    ].copy()

    candidates = prepare_candidates(
        run,
        run_id,
        max_candidates=maximum,
    )

    section_label("Candidate preview")

    st.dataframe(
        candidates.head(20),
        hide_index=True,
        width="stretch",
    )

    if st.button(
        "Run Optimization",
        width="stretch",
    ):
        with st.spinner(
            "Optimizing test selection..."
        ):
            problem = create_problem(
                candidates,
                budget=budget,
                redundancy_penalty=penalty,
            )

            if algorithm == "Greedy":
                selection, result, trace = solve_greedy(
                    problem
                )
            else:
                selection, result = solve_ortools(
                    problem
                )

            selected_ids = [
                test_id
                for test_id, value in zip(
                    problem.test_ids,
                    selection,
                )
                if value == 1
            ]

            selected = run[
                run["test_id"].isin(selected_ids)
            ].copy()

        columns = st.columns(4)

        with columns[0]:
            metric_card(
                "Selected tests",
                len(selected),
                f"From {len(run)} observed tests",
            )

        with columns[1]:
            metric_card(
                "Estimated duration",
                f"{result['estimated_duration_seconds']:.2f}s",
                f"Budget: {budget}s",
            )

        with columns[2]:
            metric_card(
                "Objective value",
                f"{result['objective_value']:.4f}",
                "Risk minus redundancy penalty",
            )

        with columns[3]:
            metric_card(
                "Observed failures",
                int(selected["failed"].sum()),
                "Retrospective evaluation",
            )

        if result["feasible"]:
            st.success(
                "Estimated execution budget satisfied."
            )
        else:
            st.error(
                "Estimated execution budget exceeded."
            )

        actual = selected[
            "duration_seconds"
        ].sum()

        st.write(
            f"Observed cumulative execution cost: "
            f"{actual:.2f} seconds"
        )

        if actual > budget:
            st.warning(
                "Observed execution cost exceeded "
                "the estimated budget."
            )

        section_label("Selected test classes")

        st.dataframe(
            selected[
                [
                    "test_id",
                    "failure_score",
                    "estimated_duration_seconds",
                    "duration_seconds",
                    "failed",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Export selected tests",
            selected.to_csv(index=False),
            file_name="quantci_selected_tests.csv",
            mime="text/csv",
            width="stretch",
        )


def quantum_laboratory():
    page_heading(
        "Quantum Laboratory",
        "QUBO optimization and quantum-classical experiments.",
    )

    report = read_json(
        "benchmark/unified_benchmark_report.json"
    )

    summary = read_csv(
        "benchmark/unified_benchmark_summary.csv"
    )

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "Quantum variables",
            report.get("qubo_variables", "—"),
            "Test and slack variables",
        )

    with columns[1]:
        metric_card(
            "Candidate tests",
            report.get("candidate_tests", "—"),
            "Shared optimization instance",
        )

    with columns[2]:
        metric_card(
            "Exact objective",
            f"{report['exact_objective']:.4f}"
            if "exact_objective" in report
            else "—",
            "Classical reference",
        )

    with columns[3]:
        metric_card(
            "Benchmark records",
            report.get(
                "total_benchmark_records",
                "—",
            ),
            "Matched algorithm comparisons",
        )

    section_label("Quantum optimization pipeline")

    first, second = st.columns(2)

    with first:
        feature_card(
            "01",
            "QUBO formulation",
            "Represent test selection and execution constraints "
            "as a quadratic binary optimization problem.",
        )

        feature_card(
            "03",
            "QAOA simulation",
            "Use parameterized quantum circuits and a classical "
            "optimizer to sample candidate solutions.",
        )

    with second:
        feature_card(
            "02",
            "Ising Hamiltonian",
            "Translate binary optimization coefficients "
            "into Pauli-Z Hamiltonian terms.",
        )

        feature_card(
            "04",
            "Hybrid slack repair",
            "Recover valid slack assignments while preserving "
            "the sampled test-selection decisions.",
        )

    if not summary.empty:
        section_label("Classical versus quantum")

        figure = px.bar(
            summary,
            x="algorithm",
            y="mean_approximation_ratio",
            color="algorithm",
            color_discrete_sequence=PALETTE,
        )

        st.plotly_chart(
            style_chart(figure),
            width="stretch",
        )

        figure = px.bar(
            summary,
            x="algorithm",
            y="mean_runtime_seconds",
            color="algorithm",
            log_y=True,
            color_discrete_sequence=PALETTE,
        )

        st.plotly_chart(
            style_chart(figure),
            width="stretch",
        )

        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
        )

    st.warning(
        "These experiments use a quantum simulator. "
        "Quantum computational advantage has not been established."
    )


def research_analytics():
    page_heading(
        "Research Analytics",
        "Held-out optimization and experimental findings.",
    )

    summary = read_csv(
        "heldout/heldout_summary.csv"
    )

    if summary.empty:
        st.info(
            "Complete Step 13 to unlock held-out analytics."
        )
        return

    budget = st.selectbox(
        "Execution budget",
        sorted(
            summary["budget_seconds"].unique()
        ),
    )

    filtered = summary[
        summary["budget_seconds"] == budget
    ]

    columns = st.columns(3)

    with columns[0]:
        metric_card(
            "Evaluated strategies",
            filtered["algorithm"].nunique(),
            "Shared ML-ranked shortlist",
        )

    with columns[1]:
        metric_card(
            "Best failure recall",
            format_percent(
                filtered["failure_recall"].max()
            ),
            "Descriptive held-out result",
        )

    with columns[2]:
        metric_card(
            "Maximum test reduction",
            format_percent(
                filtered["test_reduction"].max()
            ),
            "Across evaluated strategies",
        )

    section_label("Failure detection")

    figure = px.line(
        summary,
        x="budget_seconds",
        y="failure_recall",
        color="algorithm",
        markers=True,
        color_discrete_sequence=PALETTE,
    )

    st.plotly_chart(
        style_chart(figure),
        width="stretch",
    )

    section_label("Execution cost reduction")

    figure = px.line(
        summary,
        x="budget_seconds",
        y="actual_runtime_reduction",
        color="algorithm",
        markers=True,
        color_discrete_sequence=PALETTE,
    )

    st.plotly_chart(
        style_chart(figure),
        width="stretch",
    )

    section_label("Budget-specific results")

    st.dataframe(
        filtered,
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Download evaluation results",
        summary.to_csv(index=False),
        file_name="quantci_research_results.csv",
        mime="text/csv",
    )

    with st.expander("Research limitations"):
        st.markdown(
            """
            - The chronological test partition contains few failures.
            - Test rosters are retrospectively observed.
            - Risk scores are not calibrated probabilities.
            - Execution costs represent cumulative serial duration.
            - Estimated budget feasibility does not guarantee actual feasibility.
            - Quantum experiments cover a small validation instance.
            - Quantum advantage has not been established.
            """
        )


with st.sidebar:
    brand()

    st.write("")

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Dataset Intelligence",
            "ML Intelligence",
            "Test Optimizer",
            "Quantum Laboratory",
            "Research Analytics",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("RESEARCH ENVIRONMENT")

    st.markdown(
        """
        <div class="status-pill">
            <span class="status-dot"></span>
            LOCAL WORKSPACE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.caption(
        "QuantCI / Research Edition 1.0"
    )

    st.caption(
        "Machine Learning · Optimization · Quantum Computing"
    )


PAGES = {
    "Overview": overview,
    "Dataset Intelligence": dataset_intelligence,
    "ML Intelligence": ml_intelligence,
    "Test Optimizer": test_optimizer,
    "Quantum Laboratory": quantum_laboratory,
    "Research Analytics": research_analytics,
}

PAGES[page]()

st.divider()

st.caption(
    "QUANTCI © 2026 — AI & Quantum Optimization Research Platform"
)