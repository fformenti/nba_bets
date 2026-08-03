"""Playoff standings and clinching flags feature.

Computes per-team conference standings and three boolean playoff-status flags
for every (team, game) snapshot. All stats reflect the team's record *before*
the current game (no data leakage).

Standings snapshot definition
------------------------------
For a game on date D, standings are derived from each conference team's most
recent *completed* record as of D — meaning games finished on dates < D are
included (post-game stats) while today's games have not yet been played
(pre-game stats are used for teams playing today).
"""

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _get_playoff_cutoff(season: str) -> int:
    """Return playoff spots per conference for the given season.

    1980–2019: 8 spots per conference.
    2020–present: 10 spots per conference (play-in era).
    """
    start_year = int(season.split("/")[0])
    return 8 if start_year < 2020 else 10


def _build_team_game_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Split game-level rows into one row per (team, game).

    Duplicate ``(gameId, season)`` rows are dropped (first wins) so merged
    feature tables cannot inflate per-team game counts.

    Adds:
    - ``is_conf_game``: 1 if both teams share a conference.
    - ``pts_diff``: point differential from this team's perspective.
    - ``opponentId``: the opposing team's ID.
    """
    games = games.drop_duplicates(subset=["gameId", "season"], keep="first")
    base_cols = ["gameId", "gameDate", "gameDateOnlyStr", "season", "winner"]

    home = games[
        base_cols
        + [
            "hometeamId",
            "awayteamId",
            "hometeamConference",
            "awayteamConference",
            "homeScore",
            "awayScore",
        ]
    ].copy()
    home = home.rename(
        columns={
            "hometeamId": "teamId",
            "awayteamId": "opponentId",
            "hometeamConference": "conference",
            "awayteamConference": "opp_conference",
        }
    )
    home["win_bool"] = (home["teamId"] == home["winner"]).astype(int)
    home["pts_diff"] = home["homeScore"] - home["awayScore"]

    away = games[
        base_cols
        + [
            "awayteamId",
            "hometeamId",
            "awayteamConference",
            "hometeamConference",
            "homeScore",
            "awayScore",
        ]
    ].copy()
    away = away.rename(
        columns={
            "awayteamId": "teamId",
            "hometeamId": "opponentId",
            "awayteamConference": "conference",
            "hometeamConference": "opp_conference",
        }
    )
    away["win_bool"] = (away["teamId"] == away["winner"]).astype(int)
    away["pts_diff"] = away["awayScore"] - away["homeScore"]

    combined = pd.concat([home, away], ignore_index=True)
    combined["is_conf_game"] = (
        combined["conference"] == combined["opp_conference"]
    ).astype(int)
    combined = combined.drop(
        columns=["winner", "homeScore", "awayScore", "opp_conference"]
    )
    return combined.sort_values(["teamId", "season", "gameDate", "gameId"]).reset_index(
        drop=True
    )


def _compute_cumulative_stats(team_games: pd.DataFrame) -> pd.DataFrame:
    """Compute lagged cumulative stats per (teamId, season).

    Columns prefixed with ``post_`` add the current game's result to the
    pre-game totals. These are used when a past row is looked up from a
    later date (we want post-game state, not pre-game state, for completed
    games).

    All ``pre_*`` stats (raw column names) reflect the record *before* the
    current game — no leakage.
    """
    g = team_games.sort_values(["teamId", "season", "gameDate", "gameId"]).copy()
    grp = g.groupby(["teamId", "season"])

    # Shift by 1 game to avoid leakage
    g["win_l1"] = grp["win_bool"].shift(1)
    g["is_conf_game_l1"] = grp["is_conf_game"].shift(1)
    g["pts_diff_l1"] = grp["pts_diff"].shift(1)

    # Conference-specific win/loss flags (lagged)
    g["conf_win_l1"] = g["win_l1"] * g["is_conf_game_l1"]
    g["conf_loss_l1"] = (1 - g["win_l1"]) * g["is_conf_game_l1"]

    # Pre-game cumulative totals
    g["total_wins"] = grp["win_l1"].transform(pd.Series.cumsum).fillna(0).astype(int)
    g["total_losses"] = (
        (1 - g["win_l1"])
        .groupby([g["teamId"], g["season"]])
        .expanding()
        .sum()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype(int)
    )
    g["games_played"] = g["total_wins"] + g["total_losses"]
    g["conf_wins"] = (
        grp["conf_win_l1"].transform(pd.Series.cumsum).fillna(0).astype(int)
    )
    g["conf_losses"] = (
        grp["conf_loss_l1"].transform(pd.Series.cumsum).fillna(0).astype(int)
    )
    g["cum_pts_diff"] = grp["pts_diff_l1"].transform(pd.Series.cumsum).fillna(0)

    # Post-game stats: pre-game totals + this game's result
    g["post_wins"] = g["total_wins"] + g["win_bool"]
    g["post_losses"] = g["total_losses"] + (1 - g["win_bool"])
    g["post_conf_wins"] = g["conf_wins"] + (g["win_bool"] * g["is_conf_game"])
    g["post_conf_losses"] = g["conf_losses"] + ((1 - g["win_bool"]) * g["is_conf_game"])
    g["post_cum_pts_diff"] = g["cum_pts_diff"] + g["pts_diff"]

    return g.drop(
        columns=[
            "win_l1",
            "is_conf_game_l1",
            "pts_diff_l1",
            "conf_win_l1",
            "conf_loss_l1",
        ]
    )


def _merge_games_remaining(team_games: pd.DataFrame) -> pd.DataFrame:
    """Append ``games_remaining`` and ``post_games_remaining``.

    ``total_season_games`` is the league schedule length per season (balanced
    schedule: ``2 * n_unique_games // n_teams``), not per-team row counts, so
    shortened seasons and duplicate input rows are handled consistently.
    """
    season_stats = team_games.groupby("season").agg(
        n_games=("gameId", "nunique"),
        n_teams=("teamId", "nunique"),
    )
    nt = season_stats["n_teams"].to_numpy()
    ng = season_stats["n_games"].to_numpy()
    total = np.zeros(len(season_stats), dtype=np.int64)
    mask = nt > 0
    total[mask] = (2 * ng[mask]) // nt[mask]
    season_stats["total_season_games"] = total
    g = team_games.merge(
        season_stats[["total_season_games"]].reset_index(),
        on="season",
        how="left",
    )
    g["games_remaining"] = g["total_season_games"] - g["games_played"]
    g["post_games_remaining"] = g["games_remaining"] - 1
    return g.drop(columns=["total_season_games"])


def _compute_h2h_records(team_games: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative head-to-head wins per (teamId, opponentId, season).

    Only intra-conference games count for the tiebreaker.

    Returns
    -------
    pd.DataFrame
        Columns: teamId, opponentId, gameId, season, gameDateOnlyStr,
        h2h_wins_cum (pre-game), post_h2h_wins_cum (post-game).
    """
    conf_games = team_games[team_games["is_conf_game"] == 1].copy()
    conf_games = conf_games.sort_values(["teamId", "opponentId", "season", "gameDate"])

    grp = conf_games.groupby(["teamId", "opponentId", "season"])
    conf_games["h2h_win_l1"] = grp["win_bool"].shift(1).fillna(0)
    conf_games["h2h_wins_cum"] = (
        grp["h2h_win_l1"].transform(pd.Series.cumsum).fillna(0).astype(int)
    )
    # Post-game: add today's result
    conf_games["post_h2h_wins_cum"] = (
        conf_games["h2h_wins_cum"] + conf_games["win_bool"]
    )

    return conf_games[
        [
            "teamId",
            "opponentId",
            "gameId",
            "season",
            "gameDateOnlyStr",
            "h2h_wins_cum",
            "post_h2h_wins_cum",
        ]
    ]


def break_ties(
    tied_teams_df: pd.DataFrame,
    h2h_lookup: dict,
) -> pd.DataFrame:
    """Sort tied teams using NBA tiebreaker rules.

    Priority:
    1. Head-to-head record within the tied group (only vs other tied teams).
    2. Conference win percentage.
    3. Point differential.

    Parameters
    ----------
    tied_teams_df : pd.DataFrame
        One row per tied team. Required columns:
        ``teamId``, ``conf_wins``, ``conf_losses``, ``cum_pts_diff``.
    h2h_lookup : dict
        ``{(teamId_A, teamId_B): wins_of_A_vs_B}`` for all relevant pairs.

    Returns
    -------
    pd.DataFrame
        Rows sorted best-to-worst (index reset).

    Notes
    -----
    For 3+ teams still tied after H2H, the cascade continues with conference
    win % then point differential without re-evaluating H2H within sub-groups.
    This matches standard practice and is documented as a simplification for
    multi-way ties.
    """
    if len(tied_teams_df) <= 1:
        return tied_teams_df

    tied_ids = set(tied_teams_df["teamId"].tolist())
    df = tied_teams_df.copy()

    # 1. H2H wins vs only the other tied teams (not the full conference)
    df["_h2h"] = df["teamId"].apply(
        lambda t: sum(h2h_lookup.get((t, opp), 0) for opp in tied_ids if opp != t)
    )

    # 2. Conference win %
    conf_total = df["conf_wins"] + df["conf_losses"]
    df["_conf_pct"] = (df["conf_wins"] / conf_total.replace(0, np.nan)).fillna(0.0)

    df = df.sort_values(["_h2h", "_conf_pct", "cum_pts_diff"], ascending=False)
    return df.drop(columns=["_h2h", "_conf_pct"]).reset_index(drop=True)


def _rank_conference_snapshot(
    snapshot_df: pd.DataFrame,
    h2h_lookup: dict,
    season: str,
) -> pd.DataFrame:
    """Rank teams and compute playoff flags for one conference date snapshot.

    Parameters
    ----------
    snapshot_df : pd.DataFrame
        One row per team. Required columns: ``teamId``, ``total_wins``,
        ``total_losses``, ``conf_wins``, ``conf_losses``,
        ``cum_pts_diff``, ``games_remaining``.
    h2h_lookup : dict
        ``{(teamId_A, teamId_B): h2h_wins_of_A_vs_B}`` as of this snapshot.
    season : str
        Season string (e.g. ``"2024/25"``).

    Returns
    -------
    pd.DataFrame
        ``snapshot_df`` with added columns:
        ``conf_rank``, ``games_behind_leader``, ``games_behind_above``,
        ``games_ahead_of_below``, ``clinched_playoff_berth``,
        ``eliminated_from_playoffs``, ``clinched_final_seed``.
    """
    cutoff = min(_get_playoff_cutoff(season), len(snapshot_df))

    # Primary sort: most wins first, fewest losses first
    df = snapshot_df.sort_values(
        ["total_wins", "total_losses"], ascending=[False, True]
    ).reset_index(drop=True)

    # Apply tiebreakers within each group of identically-recorded teams
    records = list(zip(df["total_wins"], df["total_losses"]))
    ranked_pieces = []
    i = 0
    while i < len(df):
        j = i + 1
        while j < len(df) and records[j] == records[i]:
            j += 1
        group = df.iloc[i:j].copy()
        if j - i > 1:
            group = break_ties(group, h2h_lookup)
        ranked_pieces.append(group)
        i = j

    ranked = pd.concat(ranked_pieces, ignore_index=True)
    ranked["conf_rank"] = range(1, len(ranked) + 1)

    # --- Games behind formula ---
    # gb = ((wins_ref - wins_team) + (losses_team - losses_ref)) / 2
    # Each win is +0.5, each loss is -0.5 in the formula.

    leader_w = int(ranked.iloc[0]["total_wins"])
    leader_l = int(ranked.iloc[0]["total_losses"])

    ranked["games_behind_leader"] = ranked.apply(
        lambda r: ((leader_w - r["total_wins"]) + (r["total_losses"] - leader_l)) / 2,
        axis=1,
    )

    # GB from the team directly above (rank - 1); 0.0 for the leader
    gb_above = [0.0]
    for k in range(1, len(ranked)):
        a = ranked.iloc[k - 1]
        c = ranked.iloc[k]
        gb_above.append(
            (
                (a["total_wins"] - c["total_wins"])
                + (c["total_losses"] - a["total_losses"])
            )
            / 2
        )
    ranked["games_behind_above"] = gb_above

    # Lead over the team directly below (rank + 1); 0.0 for last place
    lead_below = []
    for k in range(len(ranked) - 1):
        c = ranked.iloc[k]
        b = ranked.iloc[k + 1]
        lead_below.append(
            (
                (c["total_wins"] - b["total_wins"])
                + (b["total_losses"] - c["total_losses"])
            )
            / 2
        )
    lead_below.append(0.0)
    ranked["games_ahead_of_below"] = lead_below

    # --- Vectorised playoff flags ---
    wins = ranked["total_wins"].values.astype(float)
    games_rem = ranked["games_remaining"].values.astype(float)
    max_wins = wins + games_rem  # best possible final win total per team

    n = len(ranked)
    clinched = np.zeros(n, dtype=bool)
    eliminated = np.zeros(n, dtype=bool)
    clinched_seed = np.zeros(n, dtype=bool)

    for i in range(n):
        # All other teams' stats (exclude self)
        other_max = np.concatenate([max_wins[:i], max_wins[i + 1 :]])
        other_wins = np.concatenate([wins[:i], wins[i + 1 :]])

        # Clinched berth: fewer than `cutoff` other teams can possibly end
        # with more wins than this team has RIGHT NOW (conservative — ignores
        # tiebreakers so a team is only marked clinched when mathematically safe).
        clinched[i] = int((other_max > wins[i]).sum()) < cutoff

        # Eliminated: at least `cutoff` teams already have more wins than
        # this team can possibly win even by going undefeated.
        eliminated[i] = int((other_wins > max_wins[i]).sum()) >= cutoff

        if clinched[i]:
            # Seed is locked when no team below can match current wins AND
            # no team above can be overtaken (only possible if they have the
            # same current wins and a tiebreaker could flip).
            below_max = max_wins[i + 1 :]
            below_can_catch = len(below_max) > 0 and float(below_max.max()) >= wins[i]

            above_curr = wins[:i]
            # This team can overtake an above team only if max_wins >= that team's current wins
            above_can_flip = len(above_curr) > 0 and max_wins[i] >= float(
                above_curr.min()
            )

            clinched_seed[i] = not below_can_catch and not above_can_flip

    ranked["clinched_playoff_berth"] = clinched
    ranked["eliminated_from_playoffs"] = eliminated
    ranked["clinched_final_seed"] = clinched_seed

    return ranked


def compute_playoff_flags(games: pd.DataFrame) -> pd.DataFrame:
    """Compute conference standings and playoff status flags for every (team, game).

    For each game on date D, standings are built from:
    - Post-game stats for teams whose last game was on a date strictly before D.
    - Pre-game stats for teams playing on date D (their result is not yet known).

    This ensures no leakage: a team's record for game G reflects only
    results from games prior to G.

    Parameters
    ----------
    games : pd.DataFrame
        Processed games with **one row per game** (not merged team-feature tables).
        Duplicate ``(gameId, season)`` rows are dropped before processing.
        Required columns: ``gameId``, ``gameDate``,
        ``gameDateOnlyStr``, ``season``, ``hometeamId``, ``awayteamId``,
        ``winner``, ``homeScore``, ``awayScore``,
        ``hometeamConference``, ``awayteamConference``.

    Returns
    -------
    pd.DataFrame
        One row per (teamId, gameId). Columns:
        ``gameId``, ``season``, ``teamId``, ``gameDate``,
        ``conf_rank``, ``games_behind_leader``,
        ``games_behind_above``, ``games_ahead_of_below``,
        ``clinched_playoff_berth``, ``eliminated_from_playoffs``,
        ``clinched_final_seed``.
    """
    team_games = _build_team_game_rows(games)
    team_games = _compute_cumulative_stats(team_games)
    team_games = _merge_games_remaining(team_games)
    h2h_df = _compute_h2h_records(team_games)

    result_pieces = []

    # groupby drops rows with a null key, so team-games whose conference is
    # missing from the lookup table are skipped here and end up with NaN
    # standings features. Ranking them as a pseudo-conference would be worse,
    # so report the loss rather than hiding it.
    no_conference = team_games["conference"].isna()
    if no_conference.any():
        skipped = team_games.loc[no_conference]
        seasons = sorted(skipped["season"].unique())
        logger.warning(
            "Playoff standings: skipping %d team-games with no conference "
            "across %d seasons (%s..%s); these get NaN standings features.",
            int(no_conference.sum()),
            len(seasons),
            seasons[0],
            seasons[-1],
        )

    for (season, conference), conf_group in team_games.groupby(
        ["season", "conference"]
    ):
        conf_team_ids = set(int(t) for t in conf_group["teamId"].unique())

        # Pre-initialise all teams at 0-0 with full games_remaining.
        # This handles teams that haven't played yet at any given date snapshot.
        init_rows = (
            conf_group.sort_values(["teamId", "gameDateOnlyStr", "gameDate"])
            .groupby("teamId", as_index=False)
            .first()
        )
        current_stats: dict = {
            int(row.teamId): {
                "total_wins": 0,
                "total_losses": 0,
                "conf_wins": 0,
                "conf_losses": 0,
                "cum_pts_diff": 0.0,
                # games_remaining for the first game = total season games (games_played=0)
                "games_remaining": int(row.games_remaining),
            }
            for row in init_rows.itertuples()
        }
        current_h2h: dict = {}

        # H2H rows for teams in this conference (already within-conference by construction)
        h2h_conf = (
            h2h_df[h2h_df["teamId"].isin(conf_team_ids)]
            .sort_values(["gameDateOnlyStr"])
            .reset_index(drop=True)
        )

        # Pre-sort group for O(N) sequential scanning
        conf_sorted = conf_group.sort_values(
            ["gameDateOnlyStr", "gameDate"]
        ).reset_index(drop=True)
        conf_tuples = list(conf_sorted.itertuples())
        h2h_tuples = list(h2h_conf.itertuples())
        conf_ptr = 0
        h2h_ptr = 0

        flag_cols = [
            "teamId",
            "conf_rank",
            "games_behind_leader",
            "games_behind_above",
            "games_ahead_of_below",
            "clinched_playoff_berth",
            "eliminated_from_playoffs",
            "clinched_final_seed",
        ]

        for date in sorted(conf_group["gameDateOnlyStr"].unique()):
            # ── Advance past games strictly before today ──────────────────────
            # Update current_stats with post-game stats so they reflect the
            # completed game result when referenced from a later date.
            while (
                conf_ptr < len(conf_tuples)
                and conf_tuples[conf_ptr].gameDateOnlyStr < date
            ):
                row = conf_tuples[conf_ptr]
                current_stats[int(row.teamId)] = {
                    "total_wins": int(row.post_wins),
                    "total_losses": int(row.post_losses),
                    "conf_wins": int(row.post_conf_wins),
                    "conf_losses": int(row.post_conf_losses),
                    "cum_pts_diff": float(row.post_cum_pts_diff),
                    "games_remaining": int(row.post_games_remaining),
                }
                conf_ptr += 1

            while (
                h2h_ptr < len(h2h_tuples) and h2h_tuples[h2h_ptr].gameDateOnlyStr < date
            ):
                row = h2h_tuples[h2h_ptr]
                current_h2h[(int(row.teamId), int(row.opponentId))] = int(
                    row.post_h2h_wins_cum
                )
                h2h_ptr += 1

            # ── Today's games: use pre-game stats ────────────────────────────
            games_today = conf_sorted[conf_sorted["gameDateOnlyStr"] == date]
            today_override: dict = {}
            for row in games_today.itertuples():
                tid = int(row.teamId)
                today_override[tid] = {
                    "total_wins": int(row.total_wins),
                    "total_losses": int(row.total_losses),
                    "conf_wins": int(row.conf_wins),
                    "conf_losses": int(row.conf_losses),
                    "cum_pts_diff": float(row.cum_pts_diff),
                    "games_remaining": int(row.games_remaining),
                }

            # Snapshot: post-game state for past teams, pre-game for today's teams
            combined = {**current_stats, **today_override}
            snapshot_df = pd.DataFrame(
                [{"teamId": tid, **stats} for tid, stats in combined.items()]
            )

            ranked = _rank_conference_snapshot(snapshot_df, current_h2h, season)

            today_out = games_today[["gameId", "season", "teamId", "gameDate"]].merge(
                ranked[flag_cols], on="teamId", how="left"
            )
            result_pieces.append(today_out)

    if not result_pieces:
        return pd.DataFrame(
            columns=[
                "gameId",
                "season",
                "teamId",
                "gameDate",
                "conf_rank",
                "games_behind_leader",
                "games_behind_above",
                "games_ahead_of_below",
                "clinched_playoff_berth",
                "eliminated_from_playoffs",
                "clinched_final_seed",
                "indifference_flag",
            ]
        )

    out = pd.concat(result_pieces, ignore_index=True)
    out["indifference_flag"] = (
        out["eliminated_from_playoffs"] | out["clinched_final_seed"]
    ).astype(int)
    return out
