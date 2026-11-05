# Games.csv corrections: NBA Cup finals and shifted tip-offs (2026-11-05)

Two defects in the raw archive `data/raw/historical/games/Games.csv`, which comes from the Kaggle
dataset and is gitignored. Both are corrected in `parse_raw_games`
(`src/etl/ingestion/raw_games.py`), not in the file itself, so re-downloading the archive cannot
bring either one back.

## 1. NBA Cup finals were labelled inconsistently

| gameId | Final | Archive `gameType` | Archive `gameSubLabel` |
|---|---|---|---|
| 62300001 | 2023 Lakers–Pacers | `NBA Cup` | blank |
| 62400001 | 2024 Thunder–Bucks | **`Regular Season`** | `Championship` |
| 62500001 | 2025 Knicks–Spurs | `in-season-knockout` | `Championship` |

The final is the only Cup game left out of the regular-season standings. Until now three special
cases kept these games out of `regular_season/games.csv`: a hardcoded `gameId != 62300001`, and
`gameSubLabel == "Championship"` filters for 2024/25 and 2025/26. Any final with a different label
would have counted towards records, streaks and SOS.

**Fix.** NBA gameIds encode the Cup final with type digit `6` (`0062…`). `GAME_ID_TYPE_PREFIX` now
maps `6` to `NBA Cup Championship`. `parse_raw_games` sets that `gameType` on every final, and
`_drop_non_regular_season_ids` drops them by gameId, which replaces the three special cases with
one rule. The regular-season table was unchanged by this (66,053 games, all three finals already
excluded).

## 2. Tip-offs 2025-11-21 to 2026-01-24 were 5 hours early

Compared against `LeagueSchedule25_26.csv` (`gameDateTimeEst`, ET wall time), 450 archive rows sit
exactly 5 h before the scheduled tip-off. Every other 2025/26 row matches. Three rows inside the
window are not shifted and must not be touched:

- 22500394 (2025-12-20 22:00): already correct.
- 62500001, the Cup final (2025-12-16 20:30): already correct.
- 22500529: a postponed fixture; the schedule shows its makeup date, 21 days later.

So a blanket shift of the window would be wrong. Every corrected time falls between 12:00 and
23:30 ET, and no correction changes the calendar date.

**Fix.** `data/raw/historical/handmade/games_tipoff_corrections.csv` (tracked) lists the 450 rows
as `gameId, archiveGameDate, gameDate`. `apply_tipoff_corrections` rewrites a row only while it
still holds the wrong `archiveGameDate`, so the fix is idempotent and turns itself off once
upstream fixes the archive. The ingest log reports how many entries no longer match.

The table was derived once from the schedule rather than at every run, because
`LEAGUE_SCHEDULE_FILE` rolls to the next season and the 2025/26 rows would lose their source. Rule
used: gameId present in the schedule, and archive time exactly 5 h before `gameDateTimeEst`.

**Impact.** After `make full-rebuild`, only `gameDate` and `gameTime` changed on those 450 rows.
Every feature value in `games_features.csv` is identical, because the shift was uniform within each
day and so preserved the slot order. The stored Polymarket odds were priced from Polymarket's own
`gameStartTime`, not our `gameDate`, so they are unaffected.

## Not fixed

12500014 (a 2025-10-06 preseason game) is stored as `20:00:00-04:00`, which fails to parse and is
dropped. It is preseason, so it never reached the regular-season table.
