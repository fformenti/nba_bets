"""Games data cleaning and transformation utilities."""

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def add_conference(games, cities_conferences, ignore_winner_column: bool = False):
    """
    Add conference information to games DataFrame.

    Each merge is validated as many-to-one: a duplicated (teamId, season) in the
    lookup table would otherwise multiply the games table by up to 8x across the
    three sequential joins, silently.

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
    lookup = cities_conferences[["teamId", "Conference", "season"]]

    games = games.merge(
        lookup,
        how="left",
        left_on=["hometeamId", "season"],
        right_on=["teamId", "season"],
        validate="m:1",
    ).drop(columns=["teamId"])

    games = games.rename(columns={"Conference": "hometeamConference"})

    games = games.merge(
        lookup,
        how="left",
        left_on=["awayteamId", "season"],
        right_on=["teamId", "season"],
        validate="m:1",
    ).drop(columns=["teamId"])
    games = games.rename(columns={"Conference": "awayteamConference"})

    if not ignore_winner_column:
        games = games.merge(
            lookup,
            how="left",
            left_on=["winner", "season"],
            right_on=["teamId", "season"],
            validate="m:1",
        ).drop(columns=["teamId"])
        games = games.rename(columns={"Conference": "winnerteamConference"})
    else:
        games["winnerteamConference"] = None

    _log_unmatched_teams(games)

    return games


def _log_unmatched_teams(games):
    """Warn about team-seasons with no conference in the lookup table."""
    unmatched = set()
    for id_col, conf_col in [
        ("hometeamId", "hometeamConference"),
        ("awayteamId", "awayteamConference"),
    ]:
        missing = games.loc[games[conf_col].isna(), [id_col, "season"]]
        unmatched.update(missing.itertuples(index=False, name=None))

    if unmatched:
        seasons = sorted({season for _, season in unmatched})
        logger.warning(
            "No conference found for %d (teamId, season) pairs across %d seasons "
            "(%s..%s); these games get NaN conference features.",
            len(unmatched),
            len(seasons),
            seasons[0],
            seasons[-1],
        )
