# Game lifecycle in the incremental pipeline

How a scheduled game becomes a history row, or never does. Every model feature reads "the games
before this one" off history's order, so the rule that matters most is the last one: **a game is
dated on the day it was played.**

## States

Each collected game is in exactly one place under `data/raw/incremental/`. So "where is the
file?" answers "what happened to the game?".

| State | Where | Gets there when | Leaves when |
|---|---|---|---|
| Pending | `upcoming_games/` | The selector finds it on the schedule and not settled in history | The results source gives a terminal status, or attempts run out |
| Played, not yet in history | `upcoming_games_results/` | Source says `final` | `append-game-results` writes it to history; the file moves to `archive/results/` |
| Postponed | `postponed/` | Source says `postponed` (also PPD or Suspended), or reports the game on a later date after its original tip-off passed | `reconcile-postponed` sees a new scheduled tip-off (date **or** time). The selector then re-emits the game on its new date |
| Cancelled | `cancelled/` | Source says `cancelled` | Never. Nothing watches it, and it is never written to history |
| Unresolved | `unresolved/` | No terminal status after `MAX_FETCH_ATTEMPTS` post-tip-off attempts, or more than `UNRESOLVED_GRACE_DAYS` behind the frontier | `make retry-unresolved`, after a human looks |

A postponed game is still owed and a cancelled one is not. If a parked postponed game drops off
the schedule, it stays parked and a warning is logged on every reconcile. A human decides whether
to move it to `cancelled/`.

## Which date is authoritative

1. **The source's played date.** `GameResult.game_date` comes from nba_api's `GAME_DATE_EST` or
   `gameEt`, from the release's schedule `game_date`, or from a placeholder file's `gameDate`. When
   it differs from the payload's date, the result payload is re-dated. It keeps the payload's time
   of day and records the old date as `scheduledGameDate`.
2. **The league schedule's date, as of the selector's last run.** This is what the pending payload
   carries. It is used only when the source gives no date.

History is sorted by `gameDate`, then `gameId`, by both writers (`ingest-raw-games` and
`append-game-results`). All feature builders sort the same way. A team plays at most once a day,
so the tie-break never orders two games of the same team.

## Rules that keep the order honest

- **A postponed history row settles nothing.** The selector still owes its fixture, and the row is
  replaced by gameId when the makeup is appended.
- **The selector's frontier is the last *played* game.** A 0-0 archive row dated past it is
  ignored. Before this fix, such a row would have quarantined every real game behind it.
- **A postponed archive row never supersedes a played history row** in `ingest-raw-games`, so a
  rebuild can't walk a makeup game back to 0-0.
- **Games from earlier seasons** that history lacks, or wrongly holds as postponed, can't be
  selected again, because the schedule covers the current season only. Put them back with
  `make requeue-games GAME_IDS="…"`, then `make fetch-game-results SOURCE=release` and
  `make append-game-results`. The release has no tip-off time. A requeued game takes the time of
  an existing history row on the same date, or midnight if there is none.

## Results sources

`SOURCE=` in the Makefile picks one: `release` (the sportsdataverse GitHub releases, the default;
see [boxscores.md](boxscores.md)), `nba_api` (stats.nba.com, which times out from the laptop) or
`placeholder` (hand-dropped files). The `release` source answers `final` with the played date,
`scheduled` for a future date, `cancelled` for a past game in a season that is over, and `unknown`
otherwise. Its data lags the games by up to a day.

`daily-cycle` re-pulls the schedule from stats.nba.com first. If that is unreachable it keeps the
schedule on disk and carries on; makeup dates published since then arrive on the next pull that
succeeds. A standalone `make fetch-league-schedule` still fails loudly.

Tests: `tests/test_postponed_lifecycle.py` runs games through the whole loop on tmp directories.
