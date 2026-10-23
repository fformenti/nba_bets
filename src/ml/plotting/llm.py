"""Interactive (plotly) figures for one LLM series, from a predictions frame.

They read the shared result schema, so they draw any series that has the
columns they need - an sklearn regressor's margins included. Moved here from
``src/ml/llm/testers.py``, which now only calls them for its notebook display.
"""

from __future__ import annotations

from itertools import accumulate

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _correct(frame: pd.DataFrame) -> list[int]:
    return (frame["y_pred"].astype(int) == frame["y_true"].astype(int)).astype(int).tolist()


def accuracy_trend_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    """Cumulative accuracy over the games in file order."""
    corrects = _correct(frame)
    x = list(range(1, len(corrects) + 1))
    running_acc = [s / i for s, i in zip(accumulate(corrects), x)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=running_acc,
            mode="lines",
            line=dict(width=3, color="firebrick"),
            name="Cumulative Accuracy",
            hovertemplate="n=%{x}<br>Accuracy=%{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.5,
        line_width=1,
        line_dash="dash",
        line_color="gray",
        annotation_text="chance (50%)",
    )
    fig.update_layout(
        title=f"{title} Accuracy: {running_acc[-1]:.1%}",
        xaxis_title="Number of Datapoints",
        yaxis_title="Accuracy",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        width=800,
        height=300,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def confidence_accuracy_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    """Accuracy among the top-k most confident games, k growing.

    Confidence is |predicted margin| when the series has one, else the distance
    of ``p_home`` from 0.5. The curve shows whether confident calls are easier
    to get right.
    """
    if frame["pred_margin"].notna().all():
        confidence, label = frame["pred_margin"].abs(), "|pred|"
        subtitle = "Sorted by |predicted point_diff| descending"
    else:
        confidence, label = (frame["p_home"] - 0.5).abs(), "|p-0.5|"
        subtitle = "Sorted by |p_home - 0.5| descending"

    order = np.argsort(-confidence.to_numpy(), kind="stable")
    corrects_ranked = np.asarray(_correct(frame))[order]
    confidence_ranked = confidence.to_numpy()[order]
    n = len(corrects_ranked)

    running_acc = np.cumsum(corrects_ranked) / np.arange(1, n + 1)
    x = np.arange(1, n + 1) / n
    overall_acc = running_acc[-1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=running_acc,
            mode="lines",
            line=dict(width=3, color="steelblue"),
            name="Accuracy @ coverage",
            customdata=list(zip(range(1, n + 1), confidence_ranked)),
            hovertemplate=(
                "Coverage=%{x:.1%}<br>"
                "Top-n=%{customdata[0]}<br>"
                f"{label}=%{{customdata[1]:.2f}}<br>"
                "Accuracy=%{y:.1%}<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=overall_acc,
        line_width=1,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"overall ({overall_acc:.1%})",
    )
    fig.add_hline(
        y=0.5,
        line_width=1,
        line_dash="dot",
        line_color="lightgray",
        annotation_text="chance (50%)",
    )
    fig.update_layout(
        title=f"{title} Confidence Curve<br><sup>{subtitle}</sup>",
        xaxis_title="Coverage (fraction of games, most confident first)",
        yaxis_title="Cumulative Accuracy",
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        width=800,
        height=400,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def margin_scatter_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    """Predicted vs actual margin, green where the sign was right."""
    df = pd.DataFrame(
        {
            "truth": frame["true_margin"].to_numpy(),
            "guess": frame["pred_margin"].to_numpy(),
            "correct": _correct(frame),
            "title": [f"game {g}" for g in frame["game_id"]],
        }
    )
    df["color"] = np.where(df["correct"] == 1, "green", "red")
    df["hover"] = [
        f"{t}\nGuess={g:+.0f} Actual={y:+.0f}\n{'Correct' if c else 'Wrong'}"
        for t, g, y, c in zip(df["title"], df["guess"], df["truth"], df["correct"])
    ]

    min_val = float(min(df["truth"].min(), df["guess"].min()))
    max_val = float(max(df["truth"].max(), df["guess"].max()))
    if min_val == max_val:
        min_val -= 1
        max_val += 1

    fig = px.scatter(
        df,
        x="truth",
        y="guess",
        color="color",
        color_discrete_map={"green": "green", "red": "red"},
        title=title,
        labels={"truth": "Actual Diff", "guess": "Predicted Diff"},
        width=800,
        height=600,
    )
    for tr in fig.data:
        mask = df["color"] == tr.name
        tr.customdata = df.loc[mask, ["hover"]].to_numpy()
        tr.hovertemplate = "%{customdata[0]}<extra></extra>"
        tr.marker.update(size=6)

    # Quadrant guides at zero (sign boundary)
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")
    fig.update_xaxes(range=[min_val, max_val])
    fig.update_yaxes(range=[min_val, max_val])
    fig.update_layout(showlegend=False)
    return fig


def series_figures(frame: pd.DataFrame, title: str) -> dict[str, go.Figure]:
    """Every figure the series has the columns for, by file stem."""
    figures = {"accuracy_trend": accuracy_trend_chart(frame, title)}
    if frame["pred_margin"].notna().all() or frame["p_home"].notna().all():
        figures["confidence_accuracy"] = confidence_accuracy_chart(frame, title)
    if frame[["pred_margin", "true_margin"]].notna().all().all():
        accuracy = float(np.mean(_correct(frame)))
        figures["scatter"] = margin_scatter_chart(
            frame, f"{title} results<br><b>Accuracy:</b> {accuracy:.1%} ({len(frame)} games)"
        )
    return figures
