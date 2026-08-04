"""The original hand-written prose prompt for a game.

Superseded as the default by the schema-driven serializers in
``src/ml/llm/serialization.py`` — :class:`Game` hardcodes ~20 raw column names
and raises ``KeyError`` the moment the feature set changes. It is kept because
adapters already trained on the Hub used this exact wording, and comparing a
new run against them requires reproducing it.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pandas as pd

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load the base-model tokenizer once, on first use.

    Deliberately not a class attribute: as one, merely importing this module
    downloaded a multi-hundred-MB tokenizer, which made every unrelated import
    of the package slow and network-dependent.
    """
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


@dataclass
class TeamFeatures:
    """Dataclass to store team-specific features"""

    games_played: int
    games_played_location: int
    record: float
    record_last_5_games: float
    record_last_13_games: float
    record_last_26_games: float
    record_location: float
    record_last_5_location: float
    record_last_13_location: float
    avg_pts_diff: float
    avg_pts_diff_last_5: float
    avg_pts_diff_last_13: float
    avg_pts_diff_last_26: float
    avg_pts_diff_location: float
    avg_pts_diff_last_5_location: float
    avg_pts_diff_last_13_location: float
    rested_days: int
    conference: str
    days_at_location: int | None = None
    days_at_home: int | None = None
    days_on_road: int | None = None


class Game:
    """
    NBA game and information regarding teams and their records and the outcome
    """

    SYSTEM_PROMPT = (
        "You are responsible for predicting the outcome of a basketball game.\n"
        "Below you can find useful information for predicting purposes. For the home team make sure you pay attention to its stats at home and for the visitor check its stats on the road.\n"
    )
    QUESTION = (
        "What is the point differential between the home team and the visiting team?"
        " Use positive values for the home team winning the match and negative values for the visiting team winning it.\n"
    )
    PROMPT_SUFFIX = "The home team finished the game with a point differential of "

    prompt: str | None = None

    def __init__(self, data):
        self.game_id = data.get("gameId")
        self.point_diff = data.get("pts_diff", 0)
        self.home_features = self._make_team_features(data, is_home=True)
        self.away_features = self._make_team_features(data, is_home=False)
        self.east_west_pct = data["east_wins_pct_L1"]
        self._outcome_str = self._format_outcome(self.point_diff)
        self.prompt: str | None = None
        self._generate_prompt()

    @staticmethod
    def _format_outcome(point_diff: int) -> str:
        """Format point differential with proper sign."""
        return f"{'+' if point_diff >= 0 else '-'}{abs(point_diff)}"

    @staticmethod
    def _make_team_features(data, is_home):
        """Create TeamFeatures dataclass from raw game data."""
        prefix = "H" if is_home else "V"
        location = "at_home" if is_home else "on_road"

        return TeamFeatures(
            games_played=data[f"games_played_{prefix}T"],
            games_played_location=data[f"games_played_{prefix}T"],
            record=data[f"record_L82{prefix}T"],
            record_last_5_games=data[f"record_L5_{prefix}T"],
            record_last_13_games=data[f"record_L13_{prefix}T"],
            record_last_26_games=data[f"record_L26_{prefix}T"],
            record_location=data[f"record_{prefix}T_{location}"],
            record_last_5_location=data[f"record_L5_{prefix}T_{location}"],
            record_last_13_location=data[f"record_L13_{prefix}T_{location}"],
            avg_pts_diff=data[f"pts_diff_avg_{prefix}T"],
            avg_pts_diff_last_5=data[f"pts_diff_avg_L5_{prefix}T"],
            avg_pts_diff_last_13=data[f"pts_diff_avg_L13_{prefix}T"],
            avg_pts_diff_last_26=data[f"pts_diff_avg_L26_{prefix}T"],
            avg_pts_diff_location=data[f"pts_diff_avg_{prefix}T_{location}"],
            avg_pts_diff_last_5_location=data[f"pts_diff_avg_L5_{prefix}T_{location}"],
            avg_pts_diff_last_13_location=data[f"pts_diff_avg_L13_{prefix}T_{location}"],
            rested_days=data[f"rested_days_{prefix}T"],
            conference=data[f"{'home' if is_home else 'away'}teamConference"],
            days_at_location=data["days_at_home"] if is_home else data["days_on_road"],
        )

    def _generate_team_description(self, features: TeamFeatures, team_type: str):
        """Generate formatted description for a team's features."""
        location = "at home" if team_type == "home" else "on the road"
        description = f"""
    {team_type} team:
        - Overall winning percentage:
            - Last {features.games_played} games: {features.record:.2f}
            - Last 5 games: {features.record_last_5_games:.2f}
            - Last 13 games: {features.record_last_13_games:.2f}
            - Last 26 games: {features.record_last_26_games:.2f}
        - Winning percentage {location}:
            - Last {features.games_played} games: {features.record_location:.2f}
            - Last 5 games: {features.record_last_5_location:.2f}
            - Last 13 games: {features.record_last_13_location:.2f}
        - Overall point differential:
            - Last {features.games_played} games: {features.avg_pts_diff:.2f}
            - Last 5 games: {features.avg_pts_diff_last_5:.2f}
            - Last 13 games: {features.avg_pts_diff_last_13:.2f}
        - Point differential {location}:
            - Last {features.games_played} games: {features.avg_pts_diff_location:.2f}
            - Last 5 games: {features.avg_pts_diff_last_5_location:.2f}
            - Last 13 games: {features.avg_pts_diff_last_13_location:.2f}
        - Has been resting for {features.rested_days} days
        - Currently {location} for {features.days_at_location} days
        - Plays in the {features.conference}ern Conference
"""

        return description

    def _generate_prompt(self) -> None:
        """Generate the complete prompt and calculate token count."""
        home_desc = self._generate_team_description(self.home_features, "home")
        away_desc = self._generate_team_description(self.away_features, "away")
        east_west_info = f"""
    The East teams' winning percentage over the West teams is {self.east_west_pct:.2f}
        """

        self.game_info_prompt = f"{home_desc}{away_desc}{east_west_info}"

        self.prompt = (
            f"{self.SYSTEM_PROMPT}"
            f"{self.game_info_prompt}\n"
            f"{self.QUESTION}"
            f"{self.PROMPT_SUFFIX} {self._outcome_str}"
        )

    @property
    def token_count(self) -> int:
        """Prompt length in base-model tokens.

        A property rather than a field set in ``__init__`` so that building a
        prompt does not require the tokenizer — only asking for its length does.
        """
        return len(get_tokenizer().encode(self.prompt, add_special_tokens=False))

    def get_test_prompt(self) -> str:
        """Return a prompt suitable for testing, with game outcome removed."""
        home_desc = self._generate_team_description(self.home_features, "home")
        away_desc = self._generate_team_description(self.away_features, "away")
        east_west_info = (
            f"The East teams' winning percentage over West teams: {self.east_west_pct:.2f}"
        )
        return f"{home_desc}\n{away_desc}\n{east_west_info}\n{self.PROMPT_SUFFIX}"

    def __repr__(self):
        """
        Return a String version of this Item
        """
        return self.game_info_prompt


def build_prose_prompt(features: pd.Series, meta: Optional[pd.Series] = None) -> str:
    """Adapter so :func:`src.ml.llm.serialization.serialize_row` can emit prose.

    Returns the outcome-free variant, so this never leaks the target.

    Raises ``KeyError`` when the frame lacks the raw ``*_HT``/``*_VT`` columns
    this template hardcodes — which is the whole reason ``markdown`` is the
    default format. The message names the missing column so the cause is
    obvious rather than surfacing as a bare KeyError deep in a dataset build.
    """
    row = dict(features)
    if meta is not None:
        row.update(dict(meta))

    try:
        return Game(row).get_test_prompt()
    except KeyError as exc:
        raise KeyError(
            f"The 'prose' prompt template requires column {exc} , which is not in "
            "this feature frame. The prose template is pinned to the original "
            "hand-written column list; use serialization_format 'markdown' or "
            "'json' for feature sets that have moved on."
        ) from exc
