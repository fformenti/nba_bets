from dataclasses import dataclass

from transformers import AutoTokenizer  # type: ignore

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"


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

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    SYSTEM_PROMPT = (
        "You are responsible for predicting the outcome of a basketball game.\n"
        "Below you can find useful information for predicting purposes. For the home team make sure you pay attention to its stats at home and for the visitor check its stats on the road.\n"
    )
    QUESTION = (
        "What is the point differential between the home team and the visiting team?"
        " Use positive values for the home team winning the match and negative values for the visiting team winning it.\n"
    )
    PROMPT_SUFFIX = "The home team finished the game with a point differential of "

    token_count: int = 0
    prompt: str | None = None

    def __init__(self, data):
        self.game_id = data["gameId"]
        self.point_diff = data["pts_diff"]
        self.home_features = self._make_team_features(data, is_home=True)
        self.away_features = self._make_team_features(data, is_home=False)
        self.east_west_pct = data["east_wins_pct_L1"]
        self._outcome_str = self._format_outcome(self.point_diff)
        self.prompt: str | None = None
        self.token_count: int = 0
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

        self.token_count = len(
            self.tokenizer.encode(self.prompt, add_special_tokens=False)
        )

    def get_test_prompt(self) -> str:
        """Return a prompt suitable for testing, with game outcome removed."""
        home_desc = self._generate_team_description(self.home_features, "home")
        away_desc = self._generate_team_description(self.away_features, "away")
        east_west_info = f"The East teams' winning percentage over West teams: {self.east_west_pct:.2f}"
        return f"{home_desc}\n{away_desc}\n{east_west_info}\n{self.PROMPT_SUFFIX}"

    def __repr__(self):
        """
        Return a String version of this Item
        """
        return self.game_info_prompt
