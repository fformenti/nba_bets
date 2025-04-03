def calculate_arena_ocuppation(home_games):
    home_games["attendance_last_game"] = home_games.groupby(["teamId"])[
        "attendance"
    ].shift(1)
    home_games["attendance_last_5"] = (
        home_games.groupby("teamId")["attendance_last_game"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    home_games["attendance_last_game"] = home_games.groupby("teamId")[
        "attendance_last_game"
    ].bfill()
    return home_games


def get_nba_season(game_date):
    """Determine the NBA season for a given date.
    NBA season spans October-June, labeled as YYYY/YY where October is in the first year.
    Example: Feb 25, 2025 → season "2024/25" (because season started Oct 2024)"""
    year = game_date.year
    month = game_date.month

    # If month is October or later, season is current_year/next_year
    if month >= 9:
        return f"{year}/{(year + 1) % 100:02d}"
    # Otherwise season is previous_year/current_year
    else:
        return f"{year - 1}/{year % 100:02d}"
