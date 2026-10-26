"""The label-based neutral-court flag.

It is the whole answer for any game without an arenaId: every upcoming game,
and every game appended by the daily loop. So it has to tell an NBA Cup group
game, played in the home team's arena, from a Cup semifinal in Las Vegas.
"""

import pandas as pd
import pytest

from src.etl.utils.common import add_neutral_court_game_flag


@pytest.mark.parametrize(
    "label, sublabel, expected",
    [
        ("Emirates NBA Cup", "East Group A", False),
        ("Emirates NBA Cup", "West Quarterfinal", False),
        ("Emirates NBA Cup", "West Semifinal", True),
        ("Emirates NBA Cup", "Championship", True),
        ("Emirates NBA Cup", None, False),
        ("NBA London Game", None, True),
        ("NBA Mexico City Game", "", True),
        ("", "", False),
        (None, None, False),
    ],
)
def test_neutral_court_flag(label, sublabel, expected):
    games = pd.DataFrame({"gameLabel": [label], "gameSubLabel": [sublabel]})

    flagged = add_neutral_court_game_flag(games)

    assert flagged["is_neutral_court_game"].iloc[0] == expected
    assert "gameLabel" not in flagged.columns


def test_cup_game_without_sublabel_column_is_not_neutral():
    games = pd.DataFrame({"gameLabel": ["Emirates NBA Cup", "NBA Paris Game"]})

    flagged = add_neutral_court_game_flag(games)

    assert flagged["is_neutral_court_game"].tolist() == [False, True]
