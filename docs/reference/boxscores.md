# Box scores

`make fetch-boxscores` downloads team and player box scores for 1996/97 onward, stores the played
games per season with the date each was played, and validates them. It only stores data: no
feature reads these files yet.

## Source

The sportsdataverse data releases on GitHub. They need no key and no scraping:
`https://github.com/sportsdataverse/sportsdataverse-data/releases/download/<tag>/<file>`.

| Tag | File | Use |
|---|---|---|
| `nba_stats_team_boxscores` | `team_boxscores_{Y}.parquet` | Stored |
| `nba_stats_player_boxscores` | `player_boxscores_{Y}.parquet` | Stored |
| `nba_stats_schedules` | `nba_schedule_{Y}.parquet` | Read, not stored: the played date and whether a game has a result |

`{Y}` is the season's end year, so 1997 is our 1996/97. Why this source and not stats.nba.com,
ESPN or Basketball-Reference, along with the terms that apply:
[2026-11-02-nba-data-sources.md](../findings/pipeline/2026-11-02-nba-data-sources.md). The
collector's docstring holds the same provenance. The same releases also back the `release` results
source ([game-lifecycle.md](game-lifecycle.md)).

## Layout

```
data/raw/boxscores/team/team_boxscores_{Y}.parquet
data/raw/boxscores/player/player_boxscores_{Y}.parquet
data/raw/boxscores/manifest.json          per file: release timestamp, download time, rows, games, games dropped
data/raw/boxscores/unplayed_games.csv     cancelled and postponed games, never stored as box scores
data/processed/boxscores_validation.csv   one row per season
```

Rows are stored as published, plus two added columns:

- **`game_date`:** the date the game was played, from the schedule. A postponed game carries its
  makeup date. Sort by it for anything rolling.
- **`season_label`:** our season format, e.g. "1996/97". The release's `season` column (the end
  year) stays.

Every season type is kept (`season_type_id`: 2 regular, 4 playoffs, 5 play-in, 6 NBA Cup final).
To join to our history, zero-pad its `gameId` to 10 characters and match on `game_id`. Nothing
here writes to the game history.

## What is stored, and what is not

Only games that were played. A game is dropped, and counted in the manifest, when:

- **it has no result:** null stats in the box file, or no W/L in the schedule. The release lists
  games before they are played: the whole remaining current season in the schedule, and
  placeholders in the box files, such as the 2026/27 Cup final a month early and last night's
  games before the next rebuild.
- **it is dated after today** (Eastern), as a second guard against future games.
- **the schedule can't date it.** None so far.

A game whose date has passed without a result goes to `unplayed_games.csv`:

| Status | Rule |
|---|---|
| `cancelled` | Its season is over. The only one on 2026-11-05 is Boston–Indiana, `0021201214`, 2013-04-16. |
| `postponed` | Current season, and the release already has results dated after it. Last night's games are not postponed, just not in yet: the release is rebuilt about daily. |

A postponed game that gets played drops out of the record on the next refresh and is stored with
its makeup date. `recorded_at` keeps the time a game was first recorded.

## Refresh rule

- **Missing season:** downloaded. The first run is the backfill (62 files, about 20 MB).
- **Current season:** downloaded again when the box releases' `timestamp.txt` differs from the one
  in the manifest. The schedule is re-read with it each time, because its own `timestamp.txt` isn't
  maintained (it said May 2026 while its 2027 file was rebuilt daily).
- **Past seasons:** left alone. The timestamp covers the whole release and moves every day the
  current season is rebuilt, so it says nothing about a past season. Run
  `make fetch-boxscores REFRESH_ALL=1` once after a season ends, or when the publisher rebuilds
  history, as it did on 2026-09-30.
- **Current season not yet published:** a 404 on the current season's file is logged and skipped,
  because a new season's file appears only once its first game is played. A 404 on any other
  season fails the run.
- `SEASONS="2025 2026"` limits a run to those end years.

## Checks

- **Schema drift is fatal and checked before anything is written.** The exact ordered columns and
  Arrow types are frozen in `src/etl/collectors/boxscores.py`, as is the requirement that each
  file holds only its own season. All 31 seasons of all three files matched on 2026-11-05.
- **Everything else is counted, not enforced:**
  - two team rows per game, one home and one away;
  - team points against the sum of player points;
  - coverage of our non-preseason history;
  - `history_date_mismatches`: games our history dates on a different day than the one they were
    played.

What the counts showed on 2026-11-05:

| Count | Value | Meaning |
|---|---|---|
| Regular-season games | Equal to the expected count in every completed season, except 2012/13 (1,229) | 1,189 through 2003/04, then 1,230; 725, 990, 1,059 and 1,080 in the short seasons. 2012/13 lacks the cancelled game |
| Games not two team rows, or without one home and one away | 0 everywhere | |
| Points mismatches | 1 in 1998/99 (`0029800661`), 1 in 1999/00 (`0029900712`) | The player rows sum 4 points away from the team total. Source errors |
| History coverage | 37,350 of 37,391 by season; all by gameId | The 41 are the 2020 bubble playoffs: history dates them 2020/21, the release files them under 2019/20 |
| History date mismatches | 1, `0022500529` | Postponed on 2026-01-08 and played on 2026-01-29, after the history froze. The daily loop collects it on its makeup date |
| Box games not in history | 5 in 2024/25 before the fix, 0 after | Makeup games from January 2025, added through `make requeue-games` on 2026-11-05 |
| Box games not in history | 630 in 2025/26, 107 in 2026/27 | Our history stops at 2026-01-26 |

The current season's team and player files are rebuilt minutes apart, so they can disagree by a
game for a few minutes. `games_team_only` and `games_player_only` count those separately.
