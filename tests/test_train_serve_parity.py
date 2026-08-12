"""The two feature paths must compute the same feature for the same game.

`make build-features` builds the feature tables from all of history. Prediction
builds the *same tables under the same filenames* from a much smaller frame — the
slate's season plus the prior one — with the slate rows standing in as
placeholders (`0-0`, `winner = 0`, no locations). Every feature is meant to be
strictly backward-looking, so the two must agree on a game that passes through
both.

Nothing checked that, and the 2026-08-11 rebuild audit found they did not agree
on 93 columns. The worst was silent and total: because only the ETL's ingestion
path enriched locations, the away team's travel distance was **zero for every
game ever predicted**, while the same feature carried real mileage in training.
Two of the three deployed models had selected it.

These tests run both paths over a synthetic mini-league, with fixture reference
tables, so they need nothing from `data/`.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.config.paths import DEFAULT_FEATURES_CONFIG_PATH
from src.etl.features.aggregator import (
    create_features_tables_from_config,
    merge_features,
)
from src.ml.config.loader import load_features_config
from src.ml.prediction.features import build_prediction_feature_base, fix_upcoming_games_cols
from src.ml.prediction.io import load_upcoming_games

# Four teams, two conferences, four cities.
TEAMS = {
    1610612738: ("East", "Boston", "MA", "Boston", "Celtics"),
    1610612752: ("East", "New York", "NY", "New York", "Knicks"),
    1610612744: ("West", "San Francisco", "CA", "Golden State", "Warriors"),
    1610612747: ("West", "Los Angeles", "CA", "Los Angeles", "Lakers"),
}
SEASONS = ["2024/25", "2025/26"]

# Straight-line-ish miles. Only distinct pairs: a team staying in its own city
# has no row here by construction, which is what makes a same-city transition
# zero travel rather than missing data.
CITY_DISTANCES = {
    ("Boston, MA", "New York, NY"): 190,
    ("Boston, MA", "San Francisco, CA"): 2700,
    ("Boston, MA", "Los Angeles, CA"): 2600,
    ("New York, NY", "San Francisco, CA"): 2570,
    ("New York, NY", "Los Angeles, CA"): 2450,
    ("San Francisco, CA", "Los Angeles, CA"): 350,
}


def _build_games() -> pd.DataFrame:
    """A double round robin per season, deterministic results.

    Two games per day, at different tip-off times, because that is what made the
    strength-of-schedule contamination possible: slate games share a calendar
    date but not a timestamp, so an early placeholder falls strictly *before* a
    late one and lands inside its opponent-lookup window. A one-game-per-day
    fixture cannot reproduce it.
    """
    a, b, c, d = list(TEAMS)
    # Six match days, two games each, four distinct teams per day — a real slate.
    # A team playing twice in one day would let its own placeholder feed its own
    # later rolling averages, which cannot happen in the league or in a slate
    # (the selector emits a single date, and duplicates are dropped upstream).
    match_days = [
        [(a, b), (c, d)],
        [(b, a), (d, c)],
        [(a, c), (b, d)],
        [(c, a), (d, b)],
        [(a, d), (b, c)],
        [(d, a), (c, b)],
    ]

    rows = []
    game_num = 1
    for season_index, season in enumerate(SEASONS):
        date = pd.Timestamp(f"20{24 + season_index}-11-01")
        for day in match_days:
            for slot, (home, away) in enumerate(day):
                # Deterministic and not degenerate: the higher id wins, except
                # every third game, so records differ across teams.
                home_wins = (home > away) != (game_num % 3 == 0)
                rows.append(
                    {
                        "gameId": int(f"00{22 + season_index}5{game_num:05d}"),
                        "gameDate": date + pd.Timedelta(hours=17 + 4 * slot),
                        "season": season,
                        "hometeamId": home,
                        "awayteamId": away,
                        "hometeamPrename": TEAMS[home][3],
                        "hometeamName": TEAMS[home][4],
                        "awayteamPrename": TEAMS[away][3],
                        "awayteamName": TEAMS[away][4],
                        "hometeamConference": TEAMS[home][0],
                        "awayteamConference": TEAMS[away][0],
                        "homeScore": 110 if home_wins else 100,
                        "awayScore": 100 if home_wins else 110,
                        "winner": home if home_wins else away,
                        "gameType": "Regular Season",
                        "overtimes": 0,
                        "postponed": 0,
                        "is_neutral_court_game": False,
                        "neutral_court": 0,
                        "arenaId": 1,
                        "attendance": 18000,
                    }
                )
                game_num += 1
            date += pd.Timedelta(days=1)

    games = pd.DataFrame(rows)
    games["gameDateOnlyStr"] = games["gameDate"].dt.strftime("%Y-%m-%d")
    games["win_bool"] = (games["hometeamId"] == games["winner"]).astype(int)
    games["pts_diff"] = games["homeScore"] - games["awayScore"]
    games["winnerteamConference"] = np.where(
        games["winner"] == games["hometeamId"],
        games["hometeamConference"],
        games["awayteamConference"],
    )
    games["hometeamLocation"] = games["hometeamId"].map(
        lambda t: f"{TEAMS[t][1]}, {TEAMS[t][2]}"
    )
    games["awayteamLocation"] = games["awayteamId"].map(
        lambda t: f"{TEAMS[t][1]}, {TEAMS[t][2]}"
    )
    games["gameLocation"] = games["hometeamLocation"]
    return games


@pytest.fixture
def league(tmp_path, monkeypatch):
    """Both feature paths, wired to fixture reference tables under tmp_path."""
    locations = pd.DataFrame(
        [
            {
                "teamId": team_id,
                "season": season,
                "city": meta[1],
                "state": meta[2],
            }
            for team_id, meta in TEAMS.items()
            for season in SEASONS
        ]
    )
    locations_path = tmp_path / "team_locations.csv"
    locations.to_csv(locations_path, index=False)

    distances = pd.DataFrame(
        [
            {
                "from": a,
                "to": b,
                "straight_line_distance": miles,
                "driving_distance": miles * 1.2,
            }
            for (a, b), miles in CITY_DISTANCES.items()
        ]
    )
    distances_path = tmp_path / "locations_distances.csv"
    distances.to_csv(distances_path, index=False)

    games = _build_games()

    # The season length must come from the schedule, not from the games present,
    # or the two paths disagree simply because one holds fewer games than the
    # other — the bug this fixture would otherwise reintroduce.
    schedule = games[["gameId", "season", "hometeamId", "awayteamId"]].rename(
        columns={"hometeamId": "homeTeamId", "awayteamId": "awayTeamId"}
    )
    schedule_path = tmp_path / "league_schedule.csv"
    schedule.to_csv(schedule_path, index=False)

    monkeypatch.setattr(
        "src.etl.utils.common.TEAMS_LOCATIONS_REFERENCE_PATH", locations_path
    )
    monkeypatch.setattr(
        "src.etl.features.distances.LOCATIONS_DISTANCES_PATH", distances_path
    )
    monkeypatch.setattr(
        "src.etl.features.playoff_standings.PROCESSED_LEAGUE_SCHEDULE_PATH",
        schedule_path,
    )
    monkeypatch.setattr(
        "src.ml.prediction.features.PREDICTION_FEATURES_DIR", tmp_path / "pred_features"
    )

    config = load_features_config(DEFAULT_FEATURES_CONFIG_PATH)

    # ── ETL path: every game, results and all ────────────────────────────────
    etl_dir = tmp_path / "etl_features"
    create_features_tables_from_config(games.copy(), config, output_dir=etl_dir)
    etl = merge_features(games.copy(), features_dir=etl_dir)

    # ── Prediction path: the final day is the slate, results withheld ────────
    slate_date = games["gameDateOnlyStr"].max()
    history = games[games["gameDateOnlyStr"] < slate_date].copy()
    slate_rows = games[games["gameDateOnlyStr"] == slate_date]

    slate_dir = tmp_path / "slate"
    slate_dir.mkdir()
    for _, row in slate_rows.iterrows():
        # Exactly the payload the collector writes: no scores, no winner, no
        # locations — the shape that made the distance bug invisible.
        payload = {
            "gameId": int(row["gameId"]),
            "gameDate": row["gameDate"].isoformat(),
            "gameDateOnlyStr": row["gameDateOnlyStr"],
            "hometeamId": int(row["hometeamId"]),
            "homeTeamCity": row["hometeamPrename"],
            "homeTeamName": row["hometeamName"],
            "hometeamConference": row["hometeamConference"],
            "awayteamId": int(row["awayteamId"]),
            "awayTeamCity": row["awayteamPrename"],
            "awayTeamName": row["awayteamName"],
            "awayteamConference": row["awayteamConference"],
            "winnerteamConference": None,
            "season": row["season"],
            "is_neutral_court_game": False,
        }
        (slate_dir / f"{int(row['gameId'])}.json").write_text(json.dumps(payload))

    upcoming = fix_upcoming_games_cols(load_upcoming_games(slate_dir))
    predicted = build_prediction_feature_base(
        upcoming_games=upcoming, historical_features=history, features_config=config
    )

    return {
        "etl": etl.set_index("gameId"),
        "predicted": predicted.set_index("gameId"),
        "slate_ids": sorted(int(g) for g in slate_rows["gameId"]),
    }


# Placeholders legitimately differ on these: they describe the played game.
EXPECTED_TO_DIFFER = {
    "homeScore", "awayScore", "winner", "win_bool", "pts_diff", "overtimes",
    "attendance", "arenaId", "gameType", "gameLabel", "gameSubLabel",
    "postponed", "winnerteamConference", "seriesGameNumber", "gameDate",
}


def test_every_feature_agrees_between_the_two_paths(league):
    """The regression that pins the whole class of bug.

    A column that disagrees is either a train/serve skew — the model was fitted
    on one signal and is served another — or a feature that saw the game's own
    outcome, since the prediction path is the same game with its result removed.
    """
    etl, predicted = league["etl"], league["predicted"]
    ids = league["slate_ids"]

    shared = [
        c
        for c in predicted.columns
        if c in etl.columns and c not in EXPECTED_TO_DIFFER
    ]
    assert len(shared) > 100, f"expected a wide comparison, got {len(shared)} columns"

    mismatched = {}
    for column in shared:
        left = predicted.loc[ids, column]
        right = etl.loc[ids, column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a, b = left.astype(float).to_numpy(), right.astype(float).to_numpy()
            differs = ~(np.isclose(a, b, rtol=0, atol=1e-9) | (np.isnan(a) & np.isnan(b)))
        else:
            differs = left.astype(str).to_numpy() != right.astype(str).to_numpy()
        if differs.any():
            mismatched[column] = int(differs.sum())

    assert not mismatched, (
        f"{len(mismatched)} feature(s) differ between the ETL and prediction "
        f"paths for the same games: {mismatched}"
    )


def test_the_two_paths_produce_the_same_columns(league):
    """Divergent column sets are how a feature ends up computed on one side only."""
    # build_prediction_feature_base drops these two on purpose: they describe the
    # outcome, which an upcoming game does not have.
    dropped_by_design = {"winner", "winnerteamConference"}

    etl_only = set(league["etl"].columns) - set(league["predicted"].columns) - dropped_by_design
    predicted_only = set(league["predicted"].columns) - set(league["etl"].columns)
    assert not etl_only, f"columns the prediction path never builds: {sorted(etl_only)}"
    assert not predicted_only, f"columns the ETL never builds: {sorted(predicted_only)}"


def test_a_slate_game_does_not_shift_its_neighbours_strength_of_schedule(league):
    """The second skew: placeholders must not count as losses for anyone.

    A not-yet-played game carries ``winner = 0``, which reads as a loss for both
    teams. Counted, it drags down the win percentage that the strength-of-schedule
    lookup serves to every later game — including the other games on the same
    slate, which tip off hours apart. The parity assertion above covers this, but
    pin it directly so a regression names itself.
    """
    ids = league["slate_ids"]
    assert len(ids) > 1, "fixture must put more than one game on the slate"

    sos_columns = [c for c in league["predicted"].columns if c.startswith("sos_L")]
    assert sos_columns, "fixture built no SOS columns"

    for column in sos_columns:
        predicted = league["predicted"].loc[ids, column].astype(float)
        actual = league["etl"].loc[ids, column].astype(float)
        assert np.allclose(
            predicted, actual, rtol=0, atol=1e-9, equal_nan=True
        ), f"{column} differs on the slate: {predicted.tolist()} vs {actual.tolist()}"


def test_away_teams_are_not_all_teleporting(league):
    """The specific failure: distance zero for the away team of every game.

    It survived because the value is *wrong*, not missing — the all-NaN guard in
    `_align_features` has nothing to catch.
    """
    predicted = league["predicted"]
    ids = league["slate_ids"]
    travelled = predicted.loc[ids, "distance_L1_VT"]

    assert travelled.notna().all(), "away travel distance is missing for the slate"
    assert (travelled > 0).any(), (
        "every away team in the slate travelled zero miles — the slate is not "
        "being enriched with locations before the distance table is built"
    )
