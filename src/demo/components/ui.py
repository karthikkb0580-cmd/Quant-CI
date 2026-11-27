from html import escape
from pathlib import Path

import streamlit as st


def load_theme():
    path = (
        Path(__file__).resolve().parent.parent
        / "styles"
        / "theme.css"
    )

    st.markdown(
        f"<style>{path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def brand():
    st.markdown(
        """
        <div class="quant-brand">
            Quant<span>CI.</span>
        </div>
        <div class="quant-subtitle">
            Intelligence Laboratory
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="status-pill">
                    <span class="status-dot"></span>
                    AI × QUANTUM RESEARCH
                </div>
                <div class="hero-title">
                    Intelligent testing.<br>
                    <span>Optimized execution.</span>
                </div>
                <div class="hero-description">
                    A research-driven platform combining machine
                    learning, constrained optimization and quantum
                    algorithms for intelligent software test selection.
                </div>
            </div>
            <div class="orbit-system">
                <div class="orbit-ring"></div>
                <div class="orbit-ring"></div>
                <div class="orbit-ring"></div>
                <div class="orbit-core"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text):
    st.markdown(
        f'<div class="section-label">{escape(str(text))}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, detail):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {escape(str(label))}
            </div>
            <div class="metric-value">
                {escape(str(value))}
            </div>
            <div class="metric-detail">
                {escape(str(detail))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(index, title, description):
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-index">
                MODULE / {escape(str(index))}
            </div>
            <div class="feature-title">
                {escape(str(title))}
            </div>
            <div class="feature-description">
                {escape(str(description))}
            </div>
            <div class="feature-visual">
                <span class="visual-node"></span>
                <span class="visual-line"></span>
                <span class="visual-node"></span>
                <span class="visual-line"></span>
                <span class="visual-node"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_heading(title, description):
    st.title(title)
    st.caption(description)