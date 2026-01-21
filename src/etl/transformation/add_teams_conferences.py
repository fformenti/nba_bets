"""Games data cleaning and transformation utilities."""


def add_conference(games, cities_conferences):
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

    games = games.merge(
        cities_conferences[["teamId", "Conference", "season"]],
        how="left",
        left_on=["winner", "season"],
        right_on=["teamId", "season"],
    ).drop(columns=["teamId"])
    games = games.rename(columns={"Conference": "winnerteamConference"}, inplace=False)

    return games
