def generate_standings(teams_record):
    # teams_record["rank"] = (
    #     teams_record.sort_values(["record", "pts_diff_avg"], ascending=[False, False])
    #     .groupby(["conference", "gameDateOnlyStr"])["record"]
    #     .rank(method="min", ascending=False)
    #     .fillna(1)
    #     .astype(int)
    # )

    # make standings table
    teams_record["rank"] = (
        (
            teams_record.sort_values(
                ["record", "pts_diff_avg"], ascending=[False, False]
            )
            .groupby(["conference", "gameDateOnlyStr"])["record"]
            .rank(method="min", ascending=False)
        )
        .fillna(1)
        .astype(int)
    )
    return teams_record
