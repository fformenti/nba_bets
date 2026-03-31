"""Metadata-based analysis: joins predictions with game metadata to surface model error patterns."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def _build_analysis_df(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Align predictions with metadata into a single DataFrame."""
    df = metadata.copy()
    df["y_true"] = y_true.values
    df["y_pred"] = y_pred
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)
    if y_pred_proba is not None:
        df["y_proba"] = y_pred_proba
    return df


# ---------------------------------------------------------------------------
# 1. Error analysis by team
# ---------------------------------------------------------------------------

def plot_error_by_team(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    top_n: int = 20,
    title: str = "Model Error Rate by Team",
    figsize: tuple = (12, 8),
) -> tuple[plt.Figure, pd.DataFrame]:
    df = _build_analysis_df(y_true, y_pred, None, metadata)

    home_stats = (
        df.groupby("hometeamName")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_home"})
    )
    away_stats = (
        df.groupby("awayteamName")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_away"})
    )

    team_stats = pd.DataFrame(index=sorted(set(home_stats.index) | set(away_stats.index)))
    team_stats["home_accuracy"] = home_stats["accuracy"]
    team_stats["home_games"] = home_stats["n_home"]
    team_stats["away_accuracy"] = away_stats["accuracy"]
    team_stats["away_games"] = away_stats["n_away"]
    team_stats = team_stats.fillna(0)
    team_stats["total_games"] = team_stats["home_games"] + team_stats["away_games"]
    team_stats["overall_accuracy"] = (
        (team_stats["home_accuracy"] * team_stats["home_games"]
         + team_stats["away_accuracy"] * team_stats["away_games"])
        / team_stats["total_games"]
    )
    team_stats["error_rate"] = 1 - team_stats["overall_accuracy"]
    team_stats = team_stats.sort_values("error_rate", ascending=False)

    plot_df = team_stats.head(top_n).sort_values("error_rate", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(plot_df)))
    bars = ax.barh(range(len(plot_df)), plot_df["error_rate"], color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index)
    ax.set_xlabel("Error Rate")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axvline(x=(1 - df["correct"].mean()), color="navy", linestyle="--", lw=1.5, label="Overall error rate")

    for bar, n in zip(bars, plot_df["total_games"].astype(int)):
        ax.annotate(f"n={n}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="dimgray")

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    return fig, team_stats.reset_index().rename(columns={"index": "team"})


# ---------------------------------------------------------------------------
# 2. Accuracy by season
# ---------------------------------------------------------------------------

def plot_accuracy_by_season(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    title: str = "Model Accuracy by Season",
    figsize: tuple = (12, 6),
) -> tuple[plt.Figure, pd.DataFrame]:
    df = _build_analysis_df(y_true, y_pred, None, metadata)

    if "season" not in df.columns:
        raise ValueError("metadata must contain a 'season' column")

    season_stats = (
        df.groupby("season")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_games"})
        .sort_index()
    )

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(season_stats.index.astype(str), season_stats["accuracy"], "o-", color="steelblue", lw=2, label="Accuracy")
    ax1.axhline(y=df["correct"].mean(), color="navy", linestyle="--", lw=1.5, label="Overall accuracy")
    ax1.set_xlabel("Season", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.bar(season_stats.index.astype(str), season_stats["n_games"], alpha=0.2, color="gray", label="Game count")
    ax2.set_ylabel("Number of Games", fontsize=12, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, season_stats.reset_index()


# ---------------------------------------------------------------------------
# 3. Conference matchup accuracy
# ---------------------------------------------------------------------------

def plot_conference_matchup_accuracy(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    title: str = "Accuracy by Conference Matchup",
    figsize: tuple = (10, 6),
) -> tuple[plt.Figure, pd.DataFrame]:
    df = _build_analysis_df(y_true, y_pred, None, metadata)

    for col in ("hometeamConference", "awayteamConference"):
        if col not in df.columns:
            raise ValueError(f"metadata must contain '{col}'")

    df["matchup"] = df["hometeamConference"] + " vs " + df["awayteamConference"]

    matchup_stats = (
        df.groupby("matchup")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_games"})
        .sort_values("accuracy", ascending=True)
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(matchup_stats)))
    bars = ax.barh(range(len(matchup_stats)), matchup_stats["accuracy"], color=colors)
    ax.set_yticks(range(len(matchup_stats)))
    ax.set_yticklabels(matchup_stats.index)
    ax.set_xlabel("Accuracy")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axvline(x=df["correct"].mean(), color="navy", linestyle="--", lw=1.5, label="Overall accuracy")

    for bar, n in zip(bars, matchup_stats["n_games"].astype(int)):
        ax.annotate(f"n={n}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    return fig, matchup_stats.reset_index()


# ---------------------------------------------------------------------------
# 4. Overtime game accuracy
# ---------------------------------------------------------------------------

def plot_overtime_accuracy(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    title: str = "Accuracy: Regular vs Overtime Games",
    figsize: tuple = (8, 6),
) -> tuple[plt.Figure, pd.DataFrame]:
    df = _build_analysis_df(y_true, y_pred, None, metadata)

    if "overtimes" not in df.columns:
        raise ValueError("metadata must contain an 'overtimes' column")

    df["game_type"] = df["overtimes"].apply(lambda x: "Overtime" if x > 0 else "Regular")

    ot_stats = (
        df.groupby("game_type")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_games"})
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2196F3", "#FF9800"]
    bars = ax.bar(ot_stats.index, ot_stats["accuracy"], color=colors[:len(ot_stats)], width=0.5, edgecolor="black", linewidth=0.5)
    ax.axhline(y=df["correct"].mean(), color="navy", linestyle="--", lw=1.5, label="Overall accuracy")

    for bar, (_, row) in zip(bars, ot_stats.iterrows()):
        ax.annotate(
            f'{row["accuracy"]:.1%}\n(n={int(row["n_games"])})',
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(1.15, ot_stats["accuracy"].max() + 0.15))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    return fig, ot_stats.reset_index()


# ---------------------------------------------------------------------------
# 5. Neutral court accuracy
# ---------------------------------------------------------------------------

def plot_neutral_court_accuracy(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    title: str = "Accuracy: Home Court vs Neutral Court",
    figsize: tuple = (8, 6),
) -> tuple[plt.Figure, pd.DataFrame]:
    df = _build_analysis_df(y_true, y_pred, None, metadata)

    if "is_neutral_court_game" not in df.columns:
        raise ValueError("metadata must contain 'is_neutral_court_game'")

    df["court_type"] = df["is_neutral_court_game"].apply(
        lambda x: "Neutral Court" if x else "Home Court"
    )

    court_stats = (
        df.groupby("court_type")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_games"})
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4CAF50", "#9C27B0"]
    bars = ax.bar(court_stats.index, court_stats["accuracy"], color=colors[:len(court_stats)], width=0.5, edgecolor="black", linewidth=0.5)
    ax.axhline(y=df["correct"].mean(), color="navy", linestyle="--", lw=1.5, label="Overall accuracy")

    for bar, (_, row) in zip(bars, court_stats.iterrows()):
        ax.annotate(
            f'{row["accuracy"]:.1%}\n(n={int(row["n_games"])})',
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(1.15, court_stats["accuracy"].max() + 0.15))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    return fig, court_stats.reset_index()


# ---------------------------------------------------------------------------
# 6. Score margin calibration
# ---------------------------------------------------------------------------

def plot_score_margin_calibration(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    metadata: pd.DataFrame,
    title: str = "Predicted Probability vs Actual Score Margin",
    figsize: tuple = (12, 7),
) -> tuple[plt.Figure, pd.DataFrame]:
    df = _build_analysis_df(y_true, None, y_pred_proba, metadata)
    # y_pred is unused for this plot, fill with zeros
    df["y_pred"] = 0

    for col in ("homeScore", "awayScore"):
        if col not in df.columns:
            raise ValueError(f"metadata must contain '{col}'")

    df["score_margin"] = pd.to_numeric(df["homeScore"], errors="coerce") - pd.to_numeric(df["awayScore"], errors="coerce")

    margin_bins = pd.cut(df["score_margin"], bins=15)
    bin_stats = (
        df.groupby(margin_bins, observed=True)
        .agg(
            mean_proba=("y_proba", "mean"),
            n_games=("y_proba", "count"),
            mean_margin=("score_margin", "mean"),
        )
        .dropna()
        .sort_values("mean_margin")
    )

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        bin_stats["mean_margin"], bin_stats["mean_proba"],
        s=bin_stats["n_games"] * 3, c=bin_stats["mean_proba"],
        cmap="RdYlGn", edgecolors="black", linewidth=0.5, alpha=0.85, vmin=0, vmax=1,
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", lw=1, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="--", lw=1, alpha=0.7)

    for _, row in bin_stats.iterrows():
        ax.annotate(f"n={int(row['n_games'])}", (row["mean_margin"], row["mean_proba"]),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7, color="dimgray")

    ax.set_xlabel("Score Margin (Home - Away)", fontsize=12)
    ax.set_ylabel("Mean Predicted Probability (Home Win)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Predicted Probability")
    plt.tight_layout()

    return fig, bin_stats.reset_index()


# ---------------------------------------------------------------------------
# 7. Accuracy by season phase (games-played buckets)
# ---------------------------------------------------------------------------

def compute_season_phase_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    bins: list[int] | None = None,
) -> pd.DataFrame:
    """Compute accuracy stratified by season phase (games-played buckets).

    Uses ``min(games_played_HT, games_played_VT)`` to assign each observation
    to a phase bucket.

    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    metadata : pd.DataFrame
        Must contain ``games_played_HT`` and ``games_played_VT``.
    bins : list[int], optional
        Bin edges. Defaults to 10-game buckets from 0 to 90.

    Returns
    -------
    pd.DataFrame
        Columns: ``phase``, ``accuracy``, ``n_games``.
    """
    if bins is None:
        bins = list(range(0, 81, 10)) + [82]

    df = _build_analysis_df(y_true, y_pred, None, metadata)

    if "games_played_HT" not in df.columns or "games_played_VT" not in df.columns:
        raise ValueError("metadata must contain 'games_played_HT' and 'games_played_VT'")

    min_gp = np.minimum(
        pd.to_numeric(df["games_played_HT"], errors="coerce"),
        pd.to_numeric(df["games_played_VT"], errors="coerce"),
    )

    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df["phase"] = pd.cut(min_gp, bins=bins, labels=labels, include_lowest=True)

    phase_stats = (
        df.groupby("phase", observed=True)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_games"})
        .reset_index()
    )
    phase_stats.rename(columns={"phase": "season_phase"}, inplace=True)
    return phase_stats


def plot_season_phase_accuracy(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    bins: list[int] | None = None,
    y_pred_baseline: Optional[np.ndarray] = None,
    title: str = "Accuracy by Season Phase (Games Played)",
    figsize: tuple = (10, 6),
) -> tuple[plt.Figure, pd.DataFrame]:
    phase_stats = compute_season_phase_metrics(y_true, y_pred, metadata, bins=bins)

    fig, ax = plt.subplots(figsize=figsize)

    x_labels = phase_stats["season_phase"].astype(str).tolist()
    x = range(len(x_labels))

    overall_acc = (y_true.values == y_pred).mean()

    ax.plot(x, phase_stats["accuracy"], "o-", color="steelblue", lw=2, ms=7, label="Model accuracy")
    ax.axhline(y=overall_acc, color="steelblue", linestyle="--", lw=1.5, alpha=0.6, label=f"Model avg ({overall_acc:.1%})")

    for xi, acc in zip(x, phase_stats["accuracy"]):
        ax.annotate(f"{acc:.1%}", (xi, acc), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="steelblue", fontweight="bold")

    baseline_stats = None
    if y_pred_baseline is not None:
        baseline_stats = compute_season_phase_metrics(y_true, y_pred_baseline, metadata, bins=bins)
        baseline_overall = (y_true.values == y_pred_baseline).mean()
        ax.plot(x, baseline_stats["accuracy"], "s--", color="tomato", lw=2, ms=7, label="Record baseline accuracy")
        ax.axhline(y=baseline_overall, color="tomato", linestyle=":", lw=1.5, alpha=0.6, label=f"Baseline avg ({baseline_overall:.1%})")
        for xi, acc in zip(x, baseline_stats["accuracy"]):
            ax.annotate(f"{acc:.1%}", (xi, acc), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8, color="tomato", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Min Games Played (HT, VT)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    y_min = phase_stats["accuracy"].min()
    if baseline_stats is not None:
        y_min = min(y_min, baseline_stats["accuracy"].min())
    ax.set_ylim(max(0, y_min - 0.08), min(1.0, phase_stats["accuracy"].max() + 0.1))

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, phase_stats


# ---------------------------------------------------------------------------
# Runner: generate all analysis outputs
# ---------------------------------------------------------------------------

def generate_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    metadata: pd.DataFrame,
    output_dir: Path,
    conference_filter: str = "all",
    model_name: str = "",
    y_pred_baseline: Optional[np.ndarray] = None,
) -> None:
    """Generate all metadata-based analysis plots and tables.

    Saves plots to ``output_dir/analysis/`` and CSV tables to ``output_dir/tables/``.
    """
    analysis_dir = output_dir / "analysis"
    tables_dir = output_dir / "tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    suffix = f" - {model_name} ({conference_filter})" if model_name else ""

    # 1. Error by team
    fig, table = plot_error_by_team(
        y_true, y_pred, metadata,
        title=f"Model Error Rate by Team{suffix}",
    )
    fig.savefig(analysis_dir / "error_by_team.png", dpi=300, bbox_inches="tight")
    table.to_csv(tables_dir / "error_by_team.csv", index=False)
    plt.close(fig)

    # 2. Accuracy by season
    if "season" in metadata.columns:
        fig, table = plot_accuracy_by_season(
            y_true, y_pred, metadata,
            title=f"Model Accuracy by Season{suffix}",
        )
        fig.savefig(analysis_dir / "accuracy_by_season.png", dpi=300, bbox_inches="tight")
        table.to_csv(tables_dir / "accuracy_by_season.csv", index=False)
        plt.close(fig)

    # 3. Conference matchup accuracy
    if "hometeamConference" in metadata.columns and "awayteamConference" in metadata.columns:
        fig, table = plot_conference_matchup_accuracy(
            y_true, y_pred, metadata,
            title=f"Accuracy by Conference Matchup{suffix}",
        )
        fig.savefig(analysis_dir / "conference_matchup_accuracy.png", dpi=300, bbox_inches="tight")
        table.to_csv(tables_dir / "conference_matchup_accuracy.csv", index=False)
        plt.close(fig)

    # 4. Overtime accuracy
    if "overtimes" in metadata.columns:
        fig, table = plot_overtime_accuracy(
            y_true, y_pred, metadata,
            title=f"Accuracy: Regular vs Overtime{suffix}",
        )
        fig.savefig(analysis_dir / "overtime_accuracy.png", dpi=300, bbox_inches="tight")
        table.to_csv(tables_dir / "overtime_accuracy.csv", index=False)
        plt.close(fig)

    # 5. Neutral court accuracy
    if "is_neutral_court_game" in metadata.columns:
        fig, table = plot_neutral_court_accuracy(
            y_true, y_pred, metadata,
            title=f"Accuracy: Home vs Neutral Court{suffix}",
        )
        fig.savefig(analysis_dir / "neutral_court_accuracy.png", dpi=300, bbox_inches="tight")
        table.to_csv(tables_dir / "neutral_court_accuracy.csv", index=False)
        plt.close(fig)

    # 7. Season phase accuracy (games-played buckets)
    if "games_played_HT" in metadata.columns and "games_played_VT" in metadata.columns:
        fig, table = plot_season_phase_accuracy(
            y_true, y_pred, metadata,
            y_pred_baseline=y_pred_baseline,
            title=f"Accuracy by Season Phase{suffix}",
        )
        fig.savefig(analysis_dir / "season_phase_accuracy.png", dpi=300, bbox_inches="tight")
        table.to_csv(tables_dir / "season_phase_accuracy.csv", index=False)
        plt.close(fig)

    # 6. Score margin calibration (requires probabilities)
    if y_pred_proba is not None and "homeScore" in metadata.columns and "awayScore" in metadata.columns:
        fig, table = plot_score_margin_calibration(
            y_true, y_pred_proba, metadata,
            title=f"Predicted Probability vs Score Margin{suffix}",
        )
        fig.savefig(analysis_dir / "score_margin_calibration.png", dpi=300, bbox_inches="tight")
        table.to_csv(tables_dir / "score_margin_calibration.csv", index=False)
        plt.close(fig)
