"""Games data cleaning and transformation utilities."""


def add_conference(games, cities_conferences, ignore_winner_column: bool = False):
    """
    Add conference information to games DataFrame.

    Parameters
    ----------
    games : pd.DataFrame
        Games DataFrame
    cities_conferences : pd.DataFrame
        Teams history DataFrame with conference information

    Returns
    -------
    pd.DataFrame
        Games DataFrame with conference columns added
    """
    games = games.merge(
        cities_conferences[["teamId", "Conference", "season"]],
        how="left",
        left_on=["hometeamId", "season"],
        right_on=["teamId", "season"],
    ).drop(columns=["teamId"])

    games = games.rename(columns={"Conference": "hometeamConference"}, inplace=False)

    games = games.merge(
        cities_conferences[["teamId", "Conference", "season"]],
        how="left",
        left_on=["awayteamId", "season"],
        right_on=["teamId", "season"],
    ).drop(columns=["teamId"])
    games = games.rename(columns={"Conference": "awayteamConference"}, inplace=False)

    if not ignore_winner_column:
        games = games.merge(
            cities_conferences[["teamId", "Conference", "season"]],
            how="left",
            left_on=["winner", "season"],
            right_on=["teamId", "season"],
        ).drop(columns=["teamId"])
        games = games.rename(
            columns={"Conference": "winnerteamConference"}, inplace=False
        )
    else:
        games["winnerteamConference"] = None

    return games
