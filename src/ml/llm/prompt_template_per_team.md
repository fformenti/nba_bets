# Fine-tuning prompt template: per-team variant

The `per_team` prompt layout (`src/ml/llm/serialization.py`), feature set and
team/delta mode controlled by `configs/llm_prompt/per_team.yaml`. In team mode
(the example below), each feature shows a Home row and an Away row. In delta
mode, a feature instead collapses to a single Home − Away row — e.g. instead
of two point-differential rows, one delta row. Both modes are produced by the
same serializer; this file documents the team-mode example.

## Prompt

```
Predict the final NBA score margin (home − away) from pre-game stats.
Adj Win% = win percentage adjusted for strength of schedule, in percent. Ln = last n games.
Scaled Margin = team's avg point differential in hundredths of the season's margin standard deviation, so 100 = one std (positive = outscored opponents).

Team                        Home     Away
Streak                        +3       +1
Venue streak                  +5       -3
Days at venue                  4        5
Rest days                      2        0
Back-to-back                   0        1
Playoffs                 in race  in race
Adj Last-season win%          51       71
Adj Last-season venue win%    58       44

Adj Win%  L3  L5  L8  L13  L21
Home      70  64  60   59   61
Away      31  24  27   34   32

Scaled Margin  L3   L5   L8  L13  L21
Home          10   14    9   13   17
Away           0  -87  -76  -57  -45

Adj Venue win%  L5 L10 L20 L41
Home            66  60  55  50
Away            40  45  42  44

Scaled Margin Venue  L5  L10  L20  L41
Home                   4   25   18   12
Away                 -75  -55  -40  -30

Avg km/day  1d  3d  7d 14d
Home       524 438  94   6
Away       300 100  40   3

Conference matchup this season (home conf vs away conf): 55% overall, 65% on home-conf floor

Home margin =
```

## Completion

```
+8
```

## Notes

- **Layout.** Each Scaled Margin table sits directly under its matching win% table
  (overall or venue) with the same windows, so a team's record and margin for the same
  window line up. The venue pair (Adj Venue win% / Scaled Margin Venue) always shows the
  same number of columns, since both are bounded by the same venue games-played count.
- **Placeholder values.** Home − Away in each column is illustrative only; the only
  requirement is internal consistency within one example (both windows of a pair, and the
  team/delta renders of the same game, describe the same underlying game).
- **Trade-off vs delta mode.** Per-team values show each team's absolute level
  (two strong teams vs two weak teams with the same gap) and cost a few more tokens. Delta
  mode is shorter and already oriented so that positive favours home. Compare both on the
  same test split (toggle via `configs/llm_prompt/per_team.yaml`).
