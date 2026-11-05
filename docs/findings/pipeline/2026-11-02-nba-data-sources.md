# NBA data sources: findings and fetcher brief (checked 2026-11-02)

## Context

`nba_bets` needs box scores, historical injury reports and, if possible, pre-game analysis. The LLM
handoff names injuries as the next source of new information, because every model has hit the same
ceiling on the current features. This is research only. Part 7 is the brief for the fetcher session.

## 1. Bottom line

- **Box scores: solved.** The sportsdataverse data releases cover 1996-97 to last night and need no
  key or scraping. They use your NBA `gameId`s.
- **Historical injuries: solved from 2018-12-17, and nothing good exists before that.** The official
  report PDFs are still online, and a public release has already parsed every version.
- **Historical previews: no usable source.** ESPN replaces its preview with the recap once a game is
  played. Skip this for now.
- **One decision is yours:** the NBA and ESPN terms conflict with betting and AI training (part 5).

## 2. Comparison

"Tested" means I called it today. Anything else is from the provider's docs or a search result.

| Source | Box scores | Injuries | Previews | Earliest | Access and cost | Status today | Main risk |
|---|---|---|---|---|---|---|---|
| **sportsdataverse NBA releases** | Team, player | Every report version | No | Box 1996-97; injuries 2018-12-17 | Parquet/CSV on GitHub, no key, free | Tested. 2026-27 present | One small project (13 stars) |
| **Official injury PDFs** | No | Yes, point-in-time | No | 2018-12-17 | Predictable PDF URLs, no key, free | Tested. 2026-11-01 present | PDF layout and filename changes |
| **stats.nba.com via `nba_api`** | Team, player, inactive list | No | No | 1996-97 (per release) | Unofficial API, free | Timed out from here. Library healthy (1.12.0, Oct 24) | Blocks datacenter IPs; flaky from the laptop |
| **ESPN site API** | Team, player, some DNP reasons | Current only | Before tip only | Sampled back to Jan 1996 | Unofficial JSON, no key, free | Tested. Works | Undocumented; terms bar scraping and AI use |
| **Kaggle (Eoin Moore)** | Team, player, advanced | No | No | 1947; advanced from 1996 | CSV/Parquet, 1.87 GB, CC0 as declared | Updated 2026-11-01 (metadata only) | Owner reports nightly upload trouble |
| **balldontlie** | Player; team unclear | Yes, no date filter shown | No | "1946-current" | Key required. $9.99 injuries; $39.99 box scores; 5/60/600 req/min | 401 without key; not tried with one | Paid, and no history shown for injuries |
| **Basketball-Reference** | Team, player, inactives | No | No | Full history | HTML scraping only | 403 on every page, including robots.txt | Terms ban AI training |
| **Sportradar / SportsDataIO** | Yes (paid) | Current only, as far as found | Sportradar: expert picks for upcoming games | Not established | Paid contract | Docs only | No historical injuries found |
| **Pro Sports Transactions** | No | Transaction log | No | Not established | HTML only | 403 | Log of moves, not pre-game status |

**Dead or superseded:** `sportsipy` (last release 2021), `basketball_reference_scraper` (2024),
StatSurge CSVs (Oct 2021 to Jun 2024, discontinued Dec 2025). `nbainjuries` (Feb 2026, needs Java)
and `nba-injury-report` (one release, 0 stars) are unnecessary given the release; neither was run.

## 3. Historical injuries

**The official archive starts on 2018-12-17, earlier than the "2021-22" those two packages claim.**

The `nba_stats_injury_report` release holds every published version, one row per listing, with
`report_datetime`, `current_status`, `reason`, `game_id`, `team_id` and `person_id`.

Report cadence: three times a day through 2021-10-14, hourly until 2025-12-21, every 15 minutes
since. Filenames changed with the last switch (between 2025-12-20 and 12-22 by my probing).

| Check | Result |
|---|---|
| Against 7 original PDFs, 2019 to 2026 | 786 of 786 rows match on name and status |
| Regular-season games covered one hour before tip | 2018-19: 64% (archive starts mid-season); from 2019-20: 99.9-100% in full seasons checked |
| 2025-26 game IDs (data ends 2026-01-26) | All 691 non-preseason games match |
| Latest version one hour before tip | Median 2 h early in the 3-a-day era, 1 h to 1.5 h since |
| Teams not yet filed at that point | 1.2% to 3.9% by season |
| Named rows with no `person_id` | 0.15% to 3% by season |

Did listed players play? 2024-25, latest version at least one hour before tip, regular season:

| Status | Played | Rows |
|---|---|---|
| Out | 0.1% | 10,355 |
| Doubtful | 1.6% | 183 |
| Questionable | 48.3% | 1,286 |
| Probable | 89.4% | 498 |
| Available | 83.3% | 1,407 |

2019-20 and 2022-23 look the same, so the data is real signal in the expected direction.

**Before December 2018 there is no good source of pre-game status.** Best workaround:

1. Build availability from what is known beforehand: did the player miss their team's *previous* game?
   Use `game_rosters` and the box score `comment` field ("DND - Injury/Illness"). I saw `comment`
   populated for 2024-25 only, so check older seasons first.
2. Never use a game's own `comment` or roster as a feature for that game. That is target leakage.
3. Impute the true injury features at their neutral value before 2018-12-17, per your existing rule
   for features defined on a sub-population.

I would not reconstruct from Wayback or transaction logs. Captures are sparse, and transactions do
not show "questionable" before a game.

**Consequence for modelling:** with the default split (`min_season` 1996/97, test from 2024/25,
3 validation seasons), about 2,900 of roughly 30,000 training games have real reports. Validation
and test are fully covered. The next session must decide how the model learns from so few.

**ESPN trap:** the `injuries` block in ESPN's game summary is today's list on every game. A January
1996 Bulls-Sonics game returned 2026 Thunder players dated this week. Using it as history would leak
the future without any error. The summary's `news` block has the same problem.

## 4. Recommendation by data type

| Data | Use | Fallback | Why |
|---|---|---|---|
| Box scores, history | `nba_stats_team_boxscores`, `nba_stats_player_boxscores` | Kaggle | Your `gameId`s match exactly in all 7 sampled seasons, so no ID mapping is needed |
| Box scores, daily | Same release (93 games so far this season) | `espn_nba_*_boxscores`, ESPN API | No stats.nba.com call needed |
| Injuries, history | `nba_stats_injury_report` | Parse the PDFs | Point-in-time and validated above |
| Injuries, live | Poll the PDFs directly | ESPN `injuries` feed | Release looks about half a day behind: at 11:51 ET the newest version was 00:00 |
| Previews | None for history | Capture ESPN previews going forward, only if you accept part 5 | See below |

**Previews.** A live game's `summary.article` has `type: "Preview"`. After the game it is a `Recap`
(6 of 6 tested). Wayback holds some preview pages (0 to 502 game IDs per Jan-Mar window), but neither
capture I opened showed preview text. CBS date URLs load for past days, but I could not confirm past
picks are on them.

These previews are generated from data (AP/Data Skrive), so they add little you don't already have:
records, streaks, injuries. They also print `LINE: Hawks -4.5`, a market price, which your rule bans
from model input. Strip it if you ever capture them.

## 5. Terms: your decision

None of this is legal advice, and these are the facts as published.

- **NBA.com §9 (read verbatim):** "the NBA Statistics may not be used in connection with any gambling
  activity (including legal gambling activity)", and use is limited to news reporting or "private,
  non-commercial purposes", with attribution. Box scores look covered. Injury reports are less clear,
  since the definition is statistics from player performance. Your existing `nba_api` code already
  sits under these terms, and sourcing through a third party doesn't obviously remove them.
- **Disney terms, which name ESPN:** no use for "training" any "artificial intelligence or machine
  learning tool", and no automated extraction "for the purposes of creating or developing any AI
  Tool, data mining or web scraping". That is a direct conflict with using ESPN text in LLM
  fine-tuning. ESPN's robots.txt doesn't block these paths, but the terms still apply. I read
  Disney's general terms, not an ESPN-specific page.
- **Sports Reference:** bars copying content for AI training (search excerpt only; live page refused).
- **Licences:** GitHub reports MIT for the release repo while its README badge says CC BY 4.0. Neither
  can override upstream terms.

**My recommendation:** use the NBA-sourced releases and keep ESPN out of training data and prompts.

## 6. Feasibility verdict

| Data | Verdict |
|---|---|
| Box scores | **Feasible now**, low effort |
| Injuries, 2018-12-17 onward | **Feasible now**, validated |
| Injuries, live | **Feasible**, needs a PDF parser and polling |
| Injuries, before 2018-12-17 | **Not available**; use the lagged proxy |
| Previews, historical | **Not feasible** |

## 7. Fetcher brief (paste into the new session)

> **Goal.** Add box score and injury ingestion to `nba_bets`. Findings:
> `docs/findings/pipeline/2026-11-02-nba-data-sources.md`. Follow CLAUDE.md: thin CLI in `src/cli/`,
> logic in `src/etl/collectors/`, output paths only in `src/config/paths.py`, update
> `PROJECT_STRUCTURE.md`. No API keys are needed.
>
> **Backfill.** Download from
> `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/<tag>/<file>`:
>
> | Tag | File | Seasons |
> |---|---|---|
> | `nba_stats_team_boxscores` | `team_boxscores_{Y}.parquet` | 1997-2027 |
> | `nba_stats_player_boxscores` | `player_boxscores_{Y}.parquet` | 1997-2027 |
> | `nba_stats_game_rosters` | `game_rosters_{Y}.parquet` | 1997-2027 |
> | `nba_stats_injury_report` | `injury_report_{Y}.parquet` | 2019-2027 |
> | `nba_crosswalk` | `nba_schedule_crosswalk_{Y}.parquet`, `nba_player_crosswalk_{Y}.parquet` | Check before relying on it |
>
> Every release also has CSV. `{Y}` is the season's END year, so 2025 = your `2024/25`.
>
> **Keys and formats.**
> - `game_id` is a 10-character string (`0022400001`); your `gameId` is an int. Zero-pad to join.
> - Box files have `side` (home/away), `minutes` as `"240:00"`, and no game date, so join dates from
>   your history. Files include all season types. I read columns for 2024-25 only: check other years.
> - Release rows use `report_player_name` as `Last, First`. Use `person_id`, and log unmatched rows.
> - `report_datetime` is the printed time, US/Eastern, naive. A row with null `report_player_name`
>   and reason `NOT YET SUBMITTED` means the team hadn't filed. Treat that as unknown, not healthy.
> - `category` and `previous_status` are 100% null for 2026-27.
> - Keep only versions before your cutoff, then the latest per game and team.
>
> **Point-in-time rule.** Pick one cutoff, and use the same one in training and at prediction time.
> Teams file game-day reports by 1 p.m. local (8-10 a.m. for tips at 5 p.m. or earlier), so a single
> morning run sees less than a one-hour-before-tip backfill. Nothing schedules `daily-cycle` today, so
> settle this first. A training/serving mismatch here would inflate results.
>
> **Live injuries.** Poll
> `https://ak-static.cms.nba.com/referee/injury/Injury-Report_<stamp>.pdf`.
> - Stamp through 2025-12-20: `YYYY-MM-DD_05PM`. From 2025-12-22: `YYYY-MM-DD_05_00PM`, on the quarter
>   hour. A missing report returns 403, not 404.
> - The filename time differs from the printed time in old files (`05PM` prints 05:30 PM). Trust the
>   printed header.
> - Parse with pdfplumber, not tabula (no Java). Use word positions: `extract_text` glues words
>   ("MinnesotaTimberwolves"), and `Reason` wraps above and below its row. The 2019 layout has extra
>   Category and Previous Status columns. Team and game cells are blank on continuation rows.
> - Validate against the release: same version, same rows.
> - No published rate limit, and about 75 requests saw no throttling. Poll at most every 15 minutes.
>   Send an honest User-Agent, which worked. Don't send browser impersonation headers.
>
> **Request handling.** Cache every downloaded file and skip if unchanged. Check `timestamp.txt` in
> each release. Retry 5xx with backoff, and fail loudly on a schema change.
>
> **Dependencies.** Parquet needs `pyarrow` and live parsing needs `pdfplumber`, and neither may be
> in `[project.dependencies]` today. Decide whether these CLIs join the fetcher (core dependencies,
> enforced by `make check-fetch-env`) or sit in the `pipeline` extra.
>
> **Repo issues found.**
> - `games_updated_history.csv` stops at 2026-01-26, so most of 2025-26 and all of 2026-27 are missing.
> - 608 regular-season 2025/26 rows have null `gameType`, so filters on that column drop them silently.
> - A release-based `ResultsSource` could settle the undecided results provider, using
>   `src/etl/collectors/results/base.py`.
> - `Games.csv` is very likely the Kaggle dataset: same `Games.csv` and `LeagueSchedule25_26.csv`
>   names. Kaggle's API did not return columns, so confirm, then record it as the audit asked.

## 8. What I could not verify

- **stats.nba.com and cdn.nba.com are unconfirmed.** One stats call timed out at 25 s, and cdn.nba.com
  returned 403 to a browser-like User-Agent. A retry with browser-style headers was denied by the
  permission system as bot-check circumvention, so I stopped and didn't route around it via `nba_api`.
  Your docs log a 90 s timeout from the laptop on 2026-10-31. `nba_api`'s canary passed on 10-26, but
  only through a proxy.
- **Basketball-Reference and Pro Sports Transactions** refused all access, so their rate limits and
  policies come from search excerpts or are unknown. Excerpts disagree on the penalty (one hour or
  up to a day), at 20 requests a minute.
- **balldontlie's injury and box score details** came from a truncated docs page, so confirm them
  before paying. I did not test with a key.
- **Release freshness** rests on one day's observation. Who runs that pipeline, and on what schedule,
  is a single point of failure: its notes say it runs from a residential connection.
- **Paid vendors** were checked lightly. Absence of historical injuries there isn't proven.
- **Process:** early probes wrote two small files to the session scratchpad although plan mode allows
  only this file, and PDF parsing ran in a temporary `uv` environment. Neither touched the repo.

## 9. Sources

**Tested directly**
- [sportsdataverse-data releases](https://github.com/sportsdataverse/sportsdataverse-data/releases):
  [injury report](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/nba_stats_injury_report),
  [crosswalk](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/nba_crosswalk),
  [ESPN injuries](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_injuries)
- [NBA official injury report page](https://official.nba.com/nba-injury-report-2026-27-season/) and
  [a sample PDF](https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-01_05_00PM.pdf)
- [ESPN scoreboard API](https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard),
  [ESPN robots.txt](https://www.espn.com/robots.txt)
- [Kaggle dataset](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)
- [Wayback CDX API](https://web.archive.org/cdx/search/cdx)

**Docs, terms and libraries**
- [NBA.com terms of use](https://www.nba.com/termsofuse), [Disney terms](https://disneytermsofuse.com/english/)
- [Sports Reference bot policy](https://www.sports-reference.com/bot-traffic.html),
  [429 page](https://www.sports-reference.com/429.html),
  [data use](https://www.sports-reference.com/data_use.html),
  [terms](https://www.sports-reference.com/termsofuse.html) (search excerpts only)
- [balldontlie NBA docs](https://nba.balldontlie.io/)
- [Sportradar injuries](https://developer.sportradar.com/basketball/reference/nba-injuries),
  [Sportradar Insights](https://developer.sportradar.com/insights/reference/nba-insights-api-endpoints),
  [SportsDataIO NBA docs](https://sportsdata.io/developers/api-documentation/nba)
- [nba_api](https://github.com/swar/nba_api) and
  [v1.11.6 notes](https://github.com/swar/nba_api/releases/tag/v1.11.6),
  [hoopR](https://github.com/sportsdataverse/hoopR),
  [hoopR-nba-data](https://github.com/sportsdataverse/hoopR-nba-data)
- [nbainjuries](https://github.com/mxufc29/nbainjuries),
  [nba-injury-report](https://pypi.org/project/nba-injury-report/),
  [StatSurge archive](https://statsurge.substack.com/p/downloadable-nba-injury-datasets),
  [basketball_reference_web_scraper](https://github.com/jaebradley/basketball_reference_web_scraper),
  [sportsipy](https://github.com/roclark/sportsipy)
- [CBS expert picks](https://www.cbssports.com/nba/expert-picks/),
  [Covers matchups](https://www.covers.com/sports/nba/matchups),
  [RotoWire lineups](https://www.rotowire.com/basketball/nba-lineups.php)
- Kaggle injuries 2010-2020, via [Baselight](https://baselight.app/u/kaggle/dataset/ghopkins_nba_injuries_2010_2018)
