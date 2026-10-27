"""The 'per_team' prompt: per-game rules and the team/delta switch.

Pins the hard-coded lag cut, the three configurable games-played thresholds,
the conference rule, and that the config — not the feature table — decides
whether a feature renders as team rows or a home − away delta.
"""

import pandas as pd
import pytest

from src.ml.llm.serialization import (
    PER_TEAM_FEATURES,
    PER_TEAM_SCALARS,
    PER_TEAM_SUFFIX,
    PER_TEAM_TABLES,
    bound_lags,
    load_prompt_config,
    serialize_row,
)

THRESHOLDS = {
    "last_season_min_games": 13,
    "last_season_venue_min_games": 7,
    "playoffs_min_games": 50,
}


def make_config(**modes) -> dict:
    features = dict.fromkeys(PER_TEAM_FEATURES, "team")
    features.update(modes)
    return {"thresholds": dict(THRESHOLDS), "features": features}


def make_raw(games=(30, 30), venue_games=(15, 15), confs=("East", "West")) -> pd.Series:
    """A raw feature-table row: home team at 0.6 on every rate, away team at 0.4."""
    row = {
        "games_played_HT": games[0],
        "games_played_VT": games[1],
        "games_played_HT_at_home": venue_games[0],
        "games_played_VT_on_road": venue_games[1],
        "hometeamConference": confs[0],
        "awayteamConference": confs[1],
        "east_record_adjusted": 0.55,
        "west_record_adjusted": 0.45,
        "east_record_at_east": 0.65,
        "west_record_at_west": 0.60,
        "home_adv_L1000": 0.55,
        # Outcome columns: present in the raw table, must never reach the prompt.
        "homeScore": 999,
        "awayScore": 111,
        "margin": 888,
    }
    for team in ("HT", "VT"):
        for flag in ("clinched_final_seed", "clinched_playoff_berth", "eliminated_from_playoffs"):
            row[f"{flag}_{team}"] = False
    for _, home_col, away_col, kind in PER_TEAM_SCALARS.values():
        if home_col is not None:
            row[home_col], row[away_col] = (3, 1) if kind != "pct" else (0.6, 0.4)
    for _, home_col, away_col, lags, *_ in PER_TEAM_TABLES.values():
        for n in lags:
            row[home_col.format(n=n)] = 0.6
            row[away_col.format(n=n)] = 0.4
    return pd.Series(row)


def render(raw, config) -> str:
    return serialize_row(raw, raw=raw, fmt="per_team", prompt_config=config)


@pytest.mark.parametrize(
    "games, expected",
    [
        (0, (3,)),
        (3, (3,)),
        (4, (3, 5)),
        (19, (3, 5, 8, 13, 21)),
        (21, (3, 5, 8, 13, 21)),
        (22, (3, 5, 8, 13, 21, 34)),
        (100, (3, 5, 8, 13, 21, 34, 55, 82)),
    ],
)
def test_bound_lags_keeps_first_window_covering_games_played(games, expected):
    assert bound_lags((3, 5, 8, 13, 21, 34, 55, 82), games) == expected


def test_lag_table_is_cut_by_the_more_played_team():
    text = render(make_raw(games=(19, 22)), make_config())
    assert "L34" in text
    assert "L55" not in text


def test_venue_tables_are_cut_by_venue_games():
    text = render(make_raw(games=(40, 40), venue_games=(4, 3)), make_config())
    venue = text[text.index("Adj Venue win%"):].splitlines()[0]
    assert venue.split() == ["Adj", "Venue", "win%", "L5"]


@pytest.mark.parametrize("games, shown", [((12, 40), True), ((13, 13), False)])
def test_last_season_record_drops_once_both_teams_reach_threshold(games, shown):
    assert ("Adj Last-season win%" in render(make_raw(games=games), make_config())) is shown


@pytest.mark.parametrize("venue_games, shown", [((6, 20), True), ((7, 7), False)])
def test_last_season_venue_record_drops_once_both_reach_threshold(venue_games, shown):
    text = render(make_raw(venue_games=venue_games), make_config())
    assert ("Adj Last-season venue win%" in text) is shown


@pytest.mark.parametrize("games, shown", [((50, 60), False), ((51, 51), True)])
def test_playoffs_shown_only_after_threshold(games, shown):
    assert ("Playoffs" in render(make_raw(games=games), make_config())) is shown


def test_thresholds_come_from_config():
    config = make_config()
    config["thresholds"]["last_season_min_games"] = 100
    assert "Adj Last-season win%" in render(make_raw(games=(60, 60)), config)


@pytest.mark.parametrize("confs, shown", [(("East", "West"), True), (("West", "West"), False)])
def test_conference_line_only_for_cross_conference_games(confs, shown):
    assert ("Conference matchup" in render(make_raw(confs=confs), make_config())) is shown


def test_conference_line_uses_home_conference_records():
    text = render(make_raw(confs=("West", "East")), make_config())
    assert "45% overall, 60% on home-conf floor" in text


def test_delta_mode_renders_one_home_minus_away_row():
    text = render(make_raw(), make_config(win_pct="delta", streak="delta"))
    table = text[text.index("\n\nAdj Win%") + 2 :].split("\n\n")[0].splitlines()
    assert [line.split()[0] for line in table[1:]] == ["Home−Away"]
    assert table[1].split()[1] == "+20"
    assert "Streak" in text.split("Home−Away")[1]


def test_unlisted_feature_is_dropped():
    config = make_config()
    del config["features"]["travel"]
    assert "km/day" not in render(make_raw(), config)


def test_outcome_columns_never_reach_the_prompt():
    text = render(make_raw(), make_config())
    for value in ("999", "111", "888"):
        assert value not in text


def test_prompt_ends_with_suffix():
    assert render(make_raw(), make_config()).endswith(PER_TEAM_SUFFIX)


def test_target_mode_changes_only_the_cue():
    """Classification's prompt is regression's with the last line swapped."""
    from src.ml.llm.target import SPECS

    raw, config = make_raw(), make_config()
    regression = render(raw, config)
    classification = serialize_row(raw, raw=raw, fmt="per_team", prompt_config=config, cue=SPECS["classification"].cue)

    assert regression.endswith(PER_TEAM_SUFFIX)
    assert classification == regression.removesuffix(PER_TEAM_SUFFIX) + "Home team"


def test_per_team_requires_raw_and_config():
    with pytest.raises(ValueError, match="per_team"):
        serialize_row(make_raw(), fmt="per_team")


def test_shipped_config_loads():
    config = load_prompt_config("configs/llm_prompt/per_team.yaml")
    assert set(config["features"]) <= set(PER_TEAM_FEATURES)


@pytest.mark.parametrize(
    "features, match",
    [({"nope": "team"}, "Unknown prompt feature"), ({"playoffs": "delta"}, "only be shown")],
)
def test_bad_config_is_rejected(tmp_path, features, match):
    import yaml

    path = tmp_path / "prompt.yaml"
    path.write_text(yaml.safe_dump({"thresholds": THRESHOLDS, "features": features}))
    with pytest.raises(ValueError, match=match):
        load_prompt_config(path)


def test_variant_configs_change_only_what_they_name():
    base = load_prompt_config("configs/llm_prompt/per_team.yaml")
    compact = load_prompt_config("configs/llm_prompt/per_team_compact.yaml")
    venue20 = load_prompt_config("configs/llm_prompt/per_team_venue20.yaml")

    assert compact == {**base, "layout": "compact"}
    assert venue20["features"] == base["features"]
    assert venue20["thresholds"] == {**base["thresholds"], "last_season_venue_min_games": 20}


def test_compact_layout_keeps_every_value_with_fewer_characters():
    raw = make_raw()
    aligned = render(raw, make_config())
    compact = render(raw, {**make_config(), "layout": "compact"})

    assert len(compact) < len(aligned)
    assert "  " not in compact
    assert compact.endswith(PER_TEAM_SUFFIX)
    # Same cells, same order: only the whitespace between them and the header differ.
    body = aligned.split("\n\n", 1)[1]
    assert compact.split("\n\n", 1)[1].split() == body.split()


def test_unknown_layout_is_rejected(tmp_path):
    import yaml

    path = tmp_path / "prompt.yaml"
    path.write_text(yaml.safe_dump({"thresholds": THRESHOLDS, "features": {}, "layout": "wide"}))
    with pytest.raises(ValueError, match="layout"):
        load_prompt_config(path)
