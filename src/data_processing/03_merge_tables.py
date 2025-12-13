import pandas as pd

from pathlib import Path
import sys

# Add project root to path to allow imports
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.local.constants import (  # noqa: E402
    LOCAL_REGULAR_SEASON_GAMES_PATH,
    LOCAL_GAMES_FEATURES_PATH,
    LOCAL_TEAMS_HOME_RECORDS_PATH,
    LOCAL_TEAMS_AWAY_RECORDS_PATH,
    LOCAL_TEAMS_RECORDS_PATH,
    LOCAL_TEAMS_HOME_PTS_DIFF_PATH,
    LOCAL_TEAMS_AWAY_PTS_DIFF_PATH,
    LOCAL_TEAMS_PTS_DIFF_PATH,
    LOCAL_EAST_WEST_RECORDS_PATH,
    LOCAL_RESTED_DAYS_PATH,
)


# ==== Teams Records ====
def get_home_team_record(games, teams_records):
    games = (
        games.merge(
            teams_records,
            how="left",
            left_on=["gameId", "hometeamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId"])
        .copy()
    )

    games = games.rename(
        columns={
            "total_wins": "total_wins_HT",
            "total_losses": "total_losses_HT",
            "record": "record_HT",
            "record_L5": "record_L5_HT",
            "record_L13": "record_L13_HT",
            "record_L26": "record_L26_HT",
            "games_played": "games_played_HT",
        }
    )

    return games


def get_away_team_record(games, teams_records):
    games = (
        games.merge(
            teams_records,
            how="left",
            left_on=["gameId", "awayteamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId", "season", "gameDate"])
        .copy()
    )

    games = games.rename(
        columns={
            "total_wins": "total_wins_VT",
            "total_losses": "total_losses_VT",
            "record": "record_VT",
            "record_L5": "record_L5_VT",
            "record_L13": "record_L13_VT",
            "record_L26": "record_L26_VT",
            "games_played": "games_played_VT",
        }
    )

    return games


def get_home_team_record_at_home(games, home_games_record):
    games = (
        games.merge(
            home_games_record,
            how="left",
            left_on=["gameId", "hometeamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId", "season", "gameDate"])
        .copy()
    )

    games = games.rename(
        columns={
            "total_wins": "total_wins_HT_at_home",
            "total_losses": "total_losses_HT_at_home",
            "record": "record_HT_at_home",
            "record_L5": "record_L5_HT_at_home",
            "record_L13": "record_L13_HT_at_home",
            "record_L26": "record_L26_HT_at_home",
            "games_played": "games_played_HT_at_home",
        }
    )

    return games


def get_away_team_records_on_road(games, away_games_record):
    games = (
        games.merge(
            away_games_record,
            how="left",
            left_on=["gameId", "awayteamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId", "season", "gameDate"])
        .copy()
    )

    games = games.rename(
        columns={
            "total_wins": "total_wins_VT_on_road",
            "total_losses": "total_losses_VT_on_road",
            "record": "record_VT_on_road",
            "record_L5": "record_L5_VT_on_road",
            "record_L13": "record_L13_VT_on_road",
            "record_L26": "record_L26_VT_on_road",
            "games_played": "games_played_VT_on_road",
        }
    )

    return games


# ==== Point Differential ====
def add_home_team_point_diff(games, teams_pts_diff):
    games = (
        games.merge(
            teams_pts_diff,
            how="left",
            left_on=["gameId", "hometeamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId", "season", "gameDate"])
        .copy()
    )

    games = games.rename(
        columns={
            "pts_diff_avg": "pts_diff_avg_HT",
            "pts_diff_avg_L5": "pts_diff_avg_L5_HT",
            "pts_diff_avg_L13": "pts_diff_avg_L13_HT",
            "pts_diff_avg_L26": "pts_diff_avg_L26_HT",
        }
    )

    return games


def add_home_team_point_diff_at_home(games, home_pts_diff):
    games = (
        games.merge(
            home_pts_diff,
            how="left",
            left_on=["gameId", "hometeamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId", "season", "gameDate"])
        .copy()
    )

    games = games.rename(
        columns={
            "pts_diff_avg": "pts_diff_avg_HT_at_home",
            "pts_diff_avg_L5": "pts_diff_avg_L5_HT_at_home",
            "pts_diff_avg_L13": "pts_diff_avg_L13_HT_at_home",
        }
    )

    return games


def add_away_team_point_diff(games, teams_pts_diff):
    games = (
        games.merge(
            teams_pts_diff,
            how="left",
            left_on=["gameId", "awayteamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId", "season", "gameDate"])
        .copy()
    )

    games = games.rename(
        columns={
            "pts_diff_avg": "pts_diff_avg_VT",
            "pts_diff_avg_L5": "pts_diff_avg_L5_VT",
            "pts_diff_avg_L13": "pts_diff_avg_L13_VT",
            "pts_diff_avg_L26": "pts_diff_avg_L26_VT",
        }
    )

    return games


def add_away_team_point_diff_on_road(games, away_pts_diff):
    games = (
        games.merge(
            away_pts_diff,
            how="left",
            left_on=["gameId", "awayteamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId", "season", "gameDate"])
        .copy()
    )

    games = games.rename(
        columns={
            "pts_diff_avg": "pts_diff_avg_VT_on_road",
            "pts_diff_avg_L5": "pts_diff_avg_L5_VT_on_road",
            "pts_diff_avg_L13": "pts_diff_avg_L13_VT_on_road",
        }
    )

    return games


# ==== Rested Days ====
def get_rested_days_home_team(games, rested_days):
    rest_home_teams_columns = [
        "gameDateOnlyStr",
        "teamId",
        "rested_days",
        "days_at_home",
    ]
    games = games.merge(
        rested_days[rest_home_teams_columns],
        how="left",
        left_on=["gameDateOnlyStr", "hometeamId"],
        right_on=["gameDateOnlyStr", "teamId"],
    ).drop(columns=["teamId"])

    return games.rename(
        columns={
            "rested_days": "rested_days_HT",
        }
    )


def get_rested_days_away_team(games, rested_days):
    rest_away_teams_columns = [
        "gameDateOnlyStr",
        "teamId",
        "rested_days",
        "days_on_road",
    ]
    games = games.merge(
        rested_days[rest_away_teams_columns],
        how="left",
        left_on=["gameDateOnlyStr", "awayteamId"],
        right_on=["gameDateOnlyStr", "teamId"],
    ).drop(columns=["teamId"])

    return games.rename(
        columns={
            "rested_days": "rested_days_VT",
        }
    )


if __name__ == "__main__":
    # -- Read Tables --------
    games = pd.read_csv(LOCAL_REGULAR_SEASON_GAMES_PATH)
    teams_home_record = pd.read_csv(LOCAL_TEAMS_HOME_RECORDS_PATH)
    teams_away_record = pd.read_csv(LOCAL_TEAMS_AWAY_RECORDS_PATH)
    teams_records = pd.read_csv(LOCAL_TEAMS_RECORDS_PATH)

    home_pts_diff = pd.read_csv(LOCAL_TEAMS_HOME_PTS_DIFF_PATH)
    away_pts_diff = pd.read_csv(LOCAL_TEAMS_AWAY_PTS_DIFF_PATH)
    teams_pts_diff = pd.read_csv(LOCAL_TEAMS_PTS_DIFF_PATH)

    east_west_record = pd.read_csv(LOCAL_EAST_WEST_RECORDS_PATH)
    rested_days = pd.read_csv(LOCAL_RESTED_DAYS_PATH)

    # -- Join Tables --------
    # Records
    teams_records = teams_records.drop(columns=["gameDate", "season", "win_bool"])
    games = get_home_team_record(games, teams_records)
    games = get_away_team_record(games, teams_records)

    teams_home_record = teams_home_record.drop(columns=["win_bool"])
    games = get_home_team_record_at_home(games, teams_home_record)

    teams_away_record = teams_away_record.drop(columns=["win_bool"])
    games = get_away_team_records_on_road(games, teams_away_record)

    # East vs West
    games = games.merge(
        east_west_record[["gameDateOnlyStr", "east_wins_pct_L1"]],
        how="left",
        on=["gameDateOnlyStr"],
    )

    # Point differential
    teams_pts_diff.drop(columns=["pts_diff"], inplace=True)
    games = add_home_team_point_diff(games, teams_pts_diff)
    games = add_away_team_point_diff(games, teams_pts_diff)

    home_pts_diff.drop(columns=["pts_diff"], inplace=True)
    games = add_home_team_point_diff_at_home(games, home_pts_diff)
    away_pts_diff.drop(columns=["pts_diff"], inplace=True)
    games = add_away_team_point_diff_on_road(games, away_pts_diff)

    # Rested days
    games = get_rested_days_home_team(games, rested_days)
    games = get_rested_days_away_team(games, rested_days)

    games = games.drop(
        columns=[
            "hometeamName",
            "awayteamCity",
            "attendance",
            "arenaId",
        ]
    ).copy()

    # Save Table
    games.to_csv(LOCAL_GAMES_FEATURES_PATH, index=False)
