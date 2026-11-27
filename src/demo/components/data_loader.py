import json
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[3]

RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR = BASE_DIR / "models"


def read_json(relative_path):
    path = RESULTS_DIR / relative_path

    if not path.exists():
        return {}

    return json.loads(
        path.read_text(encoding="utf-8")
    )


@st.cache_data(show_spinner=False)
def read_csv_cached(path, modified_time):
    return pd.read_csv(path)


def read_csv(relative_path):
    path = RESULTS_DIR / relative_path

    if not path.exists():
        return pd.DataFrame()

    return read_csv_cached(
        str(path),
        path.stat().st_mtime_ns,
    )


def show_plot(filename):
    path = RESULTS_DIR / "plots" / filename

    if path.exists():
        st.image(
            str(path),
            use_container_width=True,
        )
    else:
        st.info(f"Plot not available: {filename}")


def format_number(value):
    if value is None:
        return "—"

    return f"{int(value):,}"


def format_percent(value):
    if value is None:
        return "—"

    return f"{float(value) * 100:.2f}%"