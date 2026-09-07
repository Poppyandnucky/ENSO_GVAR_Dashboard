from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


DASHBOARD_FONT_STATE_KEY = "dashboard_font_size"
MIN_DASHBOARD_FONT_SIZE = 14
MAX_DASHBOARD_FONT_SIZE = 24


def configured_dashboard_font_size() -> int:
    """Return the static Streamlit theme size used as the session default."""
    configured = st.get_option("theme.baseFontSize")
    return int(configured) if configured else 16


def dashboard_font_size() -> int:
    """Return the current session's interface font size."""
    value = st.session_state.get(
        DASHBOARD_FONT_STATE_KEY,
        configured_dashboard_font_size(),
    )
    return max(MIN_DASHBOARD_FONT_SIZE, min(MAX_DASHBOARD_FONT_SIZE, int(value)))


def plot_font_size() -> int:
    """Keep chart text slightly more compact than surrounding interface text."""
    return max(12, dashboard_font_size() - 1)


def apply_plot_fonts(fig: go.Figure) -> go.Figure:
    """Apply the session font preference to Plotly text elements."""
    size = plot_font_size()
    fig.update_layout(
        font=dict(size=size),
        title=dict(font=dict(size=size + 2)),
        legend=dict(
            font=dict(size=size),
            title=dict(font=dict(size=size)),
        ),
    )
    fig.update_xaxes(
        title_font=dict(size=size),
        tickfont=dict(size=size),
    )
    fig.update_yaxes(
        title_font=dict(size=size),
        tickfont=dict(size=size),
    )
    for annotation in fig.layout.annotations or ():
        annotation.font = dict(size=size)
    return fig


def render_plotly_chart(fig: go.Figure, *args: Any, **kwargs: Any):
    """Render a Plotly figure using the dashboard's session font preference."""
    return st.plotly_chart(apply_plot_fonts(fig), *args, **kwargs)
