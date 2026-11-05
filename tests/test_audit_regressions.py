"""Regressions for the bugs found in the 2026-08-11 pipeline audit.

Each test here pins a behaviour that was wrong, silently, in a way nothing
downstream could detect. See docs/findings/pipeline/2026-08-11-pipeline-audit.md
for the findings themselves.
"""

import numpy as np
import pandas as pd
import pytest

from src.etl.ingestion.raw_games import (
    apply_tipoff_corrections,
    load_tipoff_corrections,
    parse_raw_games,
)
from src.etl.process_ingested_games import filter_regular_season_games
from src.ml.prediction.pipeline import _align_features

HOME, AWAY = 1610612741, 1610612748


# ── F3: the postponed mask ───────────────────────────────────────────────────


def _raw_game(game_id, home_score, away_score):
    """One row shaped like data/raw/historical/games/Games.csv."""
    return {
        "gameId": game_id,
        "gameDate": pd.Timestamp("2026-01-25"),
        "hometeamCity": "Boston",
        "hometeamName": "Celtics",
        "hometeamId": HOME,
        "awayteamCity": "Miami",
        "awayteamName": "Heat",
        "awayteamId": AWAY,
        "homeScore": home_score,
        "awayScore": away_score,
        "winner": HOME,
        "arenaId": 1,
        "attendance": 18000,
        "seriesGameNumber": 0,
        "gameType": "Regular Season",
        "gameLabel": "",
        "gameSubLabel": "",
    }


@pytest.mark.parametrize(
    "home_score, away_score, expected_postponed",
    [
        (110, 99, 0),  # played
        (0, 98, 1),  # no home points
        (110, 0, 1),  # no away points
        (np.nan, 105, 1),  # home score missing
        (100, np.nan, 1),  # away score missing
        (np.nan, np.nan, 1),  # neither reported
    ],
)
def test_missing_scores_are_flagged_postponed(
    home_score, away_score, expected_postponed, monkeypatch
):
    """A game with no score on record did not happen, and must not reach the
    regular-season table as if it had.

    This was broken by operator precedence: `|` binds tighter than `<=`, so
    `awayScore <= 0 | isna | isna` compared a score against a *boolean* and the
    two isna clauses never applied. Rows with a missing score sailed through with
    postponed = 0. A suspended or in-progress game is exactly that shape.
    """
    # Location enrichment reads a reference lookup off disk; the mask under test
    # does not depend on it.
    monkeypatch.setattr(
        "src.etl.ingestion.raw_games.enrich_games_locations", lambda df: df
    )

    parsed = parse_raw_games(pd.DataFrame([_raw_game(22500001, home_score, away_score)]))

    assert parsed["postponed"].iloc[0] == expected_postponed


# ── F1: the train/serve skew guard ───────────────────────────────────────────


def test_feature_empty_for_the_whole_slate_raises():
    """A feature NaN for *every* row is a train/serve skew, not a data gap.

    The model was fitted on a real signal and is being handed an imputed
    constant for it, permanently and invisibly. This is what made the
    last-season-record features silently useless in production.
    """
    features = pd.DataFrame(
        {"good_feature": [1.0, 2.0], "never_computable": [np.nan, np.nan]}
    )

    with pytest.raises(ValueError, match="NaN for every game"):
        _align_features(
            features, ["good_feature", "never_computable"], allow_missing=False
        )


def test_partially_missing_feature_is_left_alone():
    """Per-row gaps are ordinary — a team with no games played has no average."""
    features = pd.DataFrame({"rolling": [np.nan, 0.5], "other": [1.0, 2.0]})

    aligned = _align_features(features, ["rolling", "other"], allow_missing=False)

    assert len(aligned) == 2
    assert aligned["rolling"].isna().sum() == 1


def test_allow_missing_downgrades_the_skew_to_a_warning(caplog):
    """`allow_missing_features: true` should still say so, loudly."""
    features = pd.DataFrame({"never_computable": [np.nan, np.nan]})

    aligned = _align_features(features, ["never_computable"], allow_missing=True)

    assert len(aligned) == 2
    assert "NaN for every game" in caplog.text


@pytest.mark.parametrize("allow_missing", [False, True])
def test_feature_the_code_no_longer_builds_is_fatal(allow_missing):
    """A column absent from the built features is a stale model, not a data gap.

    allow_missing_features once downgraded this to a log line, and the deployed
    model served with 7 of its 14 features imputed for weeks.
    """
    features = pd.DataFrame({"still_built": [1.0, 2.0]})

    with pytest.raises(ValueError, match="no longer builds"):
        _align_features(features, ["still_built", "removed"], allow_missing=allow_missing)


def test_empty_slate_does_not_trip_the_guard():
    """No rows means no evidence either way — every column is trivially all-NaN."""
    features = pd.DataFrame({"a": pd.Series(dtype=float)})

    aligned = _align_features(features, ["a"], allow_missing=False)

    assert aligned.empty


# ── 2026-11-05: NBA Cup finals and shifted archive tip-offs ──────────────────
# See docs/findings/pipeline/2026-11-05-games-archive-corrections.md.

CUP_FINALS = [62300001, 62400001, 62500001]


@pytest.mark.parametrize("game_id", CUP_FINALS)
@pytest.mark.parametrize(
    "archive_game_type", ["Regular Season", "NBA Cup", "in-season-knockout", ""]
)
def test_cup_final_is_relabelled_whatever_the_archive_says(
    game_id, archive_game_type, monkeypatch
):
    """The archive filed the 2024/25 final as "Regular Season"."""
    monkeypatch.setattr(
        "src.etl.ingestion.raw_games.enrich_games_locations", lambda df: df
    )
    raw = _raw_game(game_id, 110, 99)
    raw["gameType"] = archive_game_type

    parsed = parse_raw_games(pd.DataFrame([raw]))

    assert parsed["gameType"].iloc[0] == "NBA Cup Championship"


@pytest.mark.parametrize("game_id", CUP_FINALS)
@pytest.mark.parametrize("game_type", ["Regular Season", "NBA Cup", ""])
@pytest.mark.parametrize("sub_label", ["Championship", ""])
def test_no_cup_final_reaches_the_regular_season_table(game_id, game_type, sub_label):
    """No final counts towards a record or streak, whatever its labels say."""
    season = {62300001: "2023/24", 62400001: "2024/25", 62500001: "2025/26"}[game_id]
    games = pd.DataFrame(
        [
            {
                "gameId": gid,
                "season": season,
                "gameType": gtype,
                "gameLabel": "Emirates NBA Cup",
                "gameSubLabel": label,
            }
            for gid, gtype, label in [
                (game_id, game_type, sub_label),
                (22400001, "Regular Season", "West Semifinal"),
            ]
        ]
    )

    kept = filter_regular_season_games(games)

    assert kept["gameId"].tolist() == [22400001]


def _corrections():
    return pd.DataFrame(
        {
            "gameId": [22500052],
            "archiveGameDate": [pd.Timestamp("2025-11-21 15:30")],
            "gameDate": [pd.Timestamp("2025-11-21 20:30")],
        }
    )


def test_a_shifted_archive_tipoff_is_corrected():
    games = pd.DataFrame(
        {"gameId": [22500052, 22500394], "gameDate": pd.to_datetime(
            ["2025-11-21 15:30", "2025-12-20 22:00"]
        )}
    )

    corrected = apply_tipoff_corrections(games, _corrections())

    assert corrected["gameDate"].tolist() == [
        pd.Timestamp("2025-11-21 20:30"),
        pd.Timestamp("2025-12-20 22:00"),
    ]


def test_an_already_correct_tipoff_is_not_shifted_again():
    """Once upstream fixes the archive, the correction must stop applying."""
    games = pd.DataFrame(
        {"gameId": [22500052], "gameDate": [pd.Timestamp("2025-11-21 20:30")]}
    )

    corrected = apply_tipoff_corrections(games, _corrections())

    assert corrected["gameDate"].tolist() == [pd.Timestamp("2025-11-21 20:30")]


def test_the_tracked_corrections_table_is_well_formed():
    corrections = load_tipoff_corrections()

    assert corrections["gameId"].is_unique
    shift = corrections["gameDate"] - corrections["archiveGameDate"]
    assert (shift == pd.Timedelta(hours=5)).all()
    assert (corrections["gameDate"].dt.date == corrections["archiveGameDate"].dt.date).all()
