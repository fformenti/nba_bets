def make_season_info(games_filtered):
    # Get seasons infos
    season_teams_ids = (
        games_filtered.groupby("season")["hometeamId"].unique().reset_index()
    )
    season_start_end = (
        games_filtered.groupby("season")["gameDate"].agg(["min", "max"]).reset_index()
    )

    season_start_end = season_start_end.rename(
        columns={"min": "season_start", "max": "season_end"}
    )

    season_info = season_start_end.merge(season_teams_ids, on="season")

    return season_info
