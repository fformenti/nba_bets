"""The result files every model family writes and the plotting module reads.

The schema is the contract that lets an sklearn classifier, a margin regressor
and an LLM adapter share one chart, so these tests pin what a writer may put in
it, how a second writer's rows join the first's, and that the LLM conversion
produces the same series the scoreboard was built from.
"""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.config.paths import RunPaths
from src.ml.results import (
    RunInfo,
    build_predictions,
    read_predictions,
    read_run_info,
    score_series,
    write_predictions,
    write_run_info,
)


def _series(run_id="r", model="m", split="validation", n=6, calibration="none", **kwargs):
    rng = np.random.RandomState(0)
    y_true = np.tile([0, 1], n // 2)
    p_home = np.clip(y_true * 0.4 + 0.3 + rng.uniform(-0.1, 0.1, n), 0.01, 0.99)
    return build_predictions(
        run_id=run_id,
        model=model,
        split=split,
        calibration=calibration,
        game_ids=kwargs.pop("game_ids", np.arange(100, 100 + n)),
        y_true=y_true,
        y_pred=(p_home > 0.5).astype(int),
        p_home=kwargs.pop("p_home", p_home),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"split": "train"}, "Unknown split"),
        ({"calibration": "temperature"}, "Unknown calibration"),
        ({"p_home": np.full(6, 1.5)}, "p_home"),
        ({"game_ids": [1, 1, 2, 3, 4, 5]}, "appears twice"),
    ],
)
def test_build_predictions_rejects_rows_that_break_the_schema(overrides, message):
    with pytest.raises(ValueError, match=message):
        _series(**overrides)


def test_run_info_round_trips_and_rejects_unknown_family():
    info = RunInfo(run_id="r", family="llm", task="classification", selected_model="r")
    assert RunInfo.from_dict({**info.to_dict(), "unknown_future_key": 1}) == info
    with pytest.raises(ValueError, match="family"):
        RunInfo(run_id="r", family="keras", task="classification", selected_model="r")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_probability_metrics_need_every_row_to_carry_p_home():
    full = score_series(_series())
    partial = score_series(_series(p_home=[0.6, 0.4, np.nan, 0.5, 0.5, 0.5]))

    assert {"brier_score", "log_loss", "ece", "roc_auc"} <= set(full)
    assert "brier_score" not in partial and partial["accuracy"] == full["accuracy"]


def test_margin_metrics_appear_only_with_both_margins():
    margins = dict(pred_margin=[2, -1, 3, -4, 1, -2], true_margin=[5, -3, 1, -6, -2, 4])
    scored = score_series(_series(**margins))
    assert scored["mae"] == pytest.approx(np.mean([3, 2, 2, 2, 3, 6]))
    assert "mae" not in score_series(_series(pred_margin=margins["pred_margin"]))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_second_writer_upserts_its_series_without_dropping_the_first(tmp_path):
    write_predictions(_series(model="a"), "r", "validation", tmp_path)
    write_predictions(_series(model="b"), "r", "validation", tmp_path)
    write_predictions(_series(model="a", calibration="platt"), "r", "validation", tmp_path)
    # Rewriting a series replaces it rather than duplicating its games.
    write_predictions(_series(model="a"), "r", "validation", tmp_path)

    frame = read_predictions("r", runs_dir=tmp_path)
    assert frame.groupby(["model", "calibration"]).size().to_dict() == {
        ("a", "none"): 6,
        ("a", "platt"): 6,
        ("b", "none"): 6,
    }
    metrics = json.loads(RunPaths("r", tmp_path).metrics("validation").read_text())
    assert len(metrics["rows"]) == 3


def test_metrics_json_writes_nan_as_null(tmp_path):
    write_predictions(_series(p_home=None), "r", "validation", tmp_path)
    text = RunPaths("r", tmp_path).metrics("validation").read_text()
    assert "NaN" not in text


def test_write_predictions_refuses_rows_of_another_split(tmp_path):
    with pytest.raises(ValueError, match="split"):
        write_predictions(_series(split="test"), "r", "validation", tmp_path)


# ---------------------------------------------------------------------------
# LLM conversion
# ---------------------------------------------------------------------------


def _inspection(n=40, seed=0):
    rng = np.random.RandomState(seed)
    margin = rng.choice([-9, -4, -2, 3, 5, 11], n)
    p_home = np.clip(0.55 + 0.03 * margin + rng.normal(0, 0.05, n), 0.02, 0.98)
    return pd.DataFrame(
        {
            "game_id": np.arange(n) + 22100000,
            "truth": [f"{m:+d}" for m in margin],
            "margin": margin,
            "margin_scale": 1.0,
            "expected_margin": margin * 0.5,
            "raw_output": ["+3" if p > 0.5 else "-3" for p in p_home],
            "raw_output_tokens": "+|3",
            "parsed": np.where(p_home > 0.5, 3.0, -3.0),
            "p_plus": p_home,
            "p_minus": 1 - p_home,
            "p_sign_mass": 1.0,
            "p_home": p_home,
            "top_first_tokens": "'+':0.6",
        }
    )


def test_llm_inspection_becomes_raw_and_platt_series(tmp_path):
    from src.ml.llm.results import write_inspection_results

    write_inspection_results(_inspection(), "e4", "validation", seed=42, runs_dir=tmp_path)
    write_inspection_results(_inspection(seed=1), "e4", "test", seed=42, runs_dir=tmp_path)

    frame = read_predictions("e4", runs_dir=tmp_path)
    assert sorted(set(zip(frame["split"], frame["calibration"]))) == [
        ("test", "none"),
        ("test", "platt"),
        ("validation", "none"),
        ("validation", "platt"),
    ]
    raw = frame[(frame["calibration"] == "none") & (frame["split"] == "validation")]
    assert (raw["y_true"] == (raw["true_margin"] > 0)).all()
    # Platt moves the probability only; the margin belongs to the raw series.
    assert frame.loc[frame["calibration"] == "platt", "pred_margin"].isna().all()

    info = read_run_info("e4", tmp_path)
    assert (info.family, info.selected_calibration) == ("llm", "platt")
    assert set(info.splits) == {"validation", "test"}
    assert RunPaths("e4", tmp_path).llm_summary("test").exists()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_comparison_scores_every_series_on_the_shared_games_only(tmp_path, monkeypatch):
    from src.ml.plotting import report

    def no_game_table(_ids):
        raise FileNotFoundError("no game table in tests")

    monkeypatch.setattr(report, "load_game_metadata", no_game_table)
    for run_id, ids in (("a", np.arange(100, 106)), ("b", np.arange(102, 108))):
        write_predictions(_series(run_id=run_id, model=run_id, game_ids=ids), run_id, "validation", tmp_path)
        write_run_info(
            RunInfo(run_id=run_id, family="sklearn", task="classification", selected_model=run_id),
            tmp_path,
        )

    report.build_comparison(["a", "b"], "ab", runs_dir=tmp_path, comparisons_dir=tmp_path / "cmp")

    scoreboard = pd.read_csv(tmp_path / "cmp" / "ab" / "tables" / "scoreboard.csv")
    assert scoreboard["n_games"].tolist() == [4, 4]
    assert scoreboard["n_games_own"].tolist() == [6, 6]
    assert (tmp_path / "cmp" / "ab" / "figures" / "roc_curve.png").exists()


def test_record_baseline_picks_the_better_record_and_ties_go_home():
    from src.ml.plotting.report import record_baseline_calls

    metadata = pd.DataFrame({"record_L82_HT": [0.6, 0.4, 0.5], "record_L82_VT": [0.4, 0.6, 0.5]})

    assert record_baseline_calls(metadata).tolist() == [1, 0, 1]


def test_comparison_adds_the_record_baseline_on_the_shared_games(tmp_path, monkeypatch):
    from src.ml.plotting import report

    def game_table(ids):
        # Home has the better record in every game but the last.
        records = [0.6] * (len(ids) - 1) + [0.3]
        return pd.DataFrame({"gameId": ids, "record_L82_HT": records, "record_L82_VT": 0.5})

    monkeypatch.setattr(report, "load_game_metadata", game_table)
    write_predictions(_series(run_id="a", model="a", game_ids=np.arange(100, 106)), "a", "validation", tmp_path)
    write_run_info(RunInfo(run_id="a", family="sklearn", task="classification", selected_model="a"), tmp_path)

    report.build_comparison(
        ["a"], "rec", include_record=True, runs_dir=tmp_path, comparisons_dir=tmp_path / "cmp"
    )

    scoreboard = pd.read_csv(tmp_path / "cmp" / "rec" / "tables" / "scoreboard.csv")
    record = scoreboard.set_index("series").loc["Record baseline"]
    assert record["n_games"] == 6
    assert record["home_pick_rate"] == pytest.approx(5 / 6)
    # A hard call carries no probability, so no probability metric is scored.
    assert pd.isna(record["log_loss"])
