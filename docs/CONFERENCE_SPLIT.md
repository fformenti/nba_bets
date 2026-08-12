# The conference split, measured — 2026-08-12

The pipeline used to train and deploy three models: one on same-conference
games, one on cross-conference games, and one on everything. At prediction time
each game was routed to its conference model *and* scored by the all-games
model. The reasoning was that conference strength varies year to year, so a
0.500 record means different things in the East and the West, and that games
within a conference are played more often.

It was never measured. This records the measurement and what changed because of
it.

## What the split cost

Three near-identical train configs; a three-way loop at prediction time that
loaded three models and ran the feature engineering three times; a
`(gameId, conference_filter)` prediction key; and — the reason this got
looked at — **two rows per game in `upcoming_games_predictions.csv` with no
rule anywhere for which one to bet**. `src/betting/bets.py` filtered on date
only, so every game was sized twice, from two different probabilities, each
against a full `GAME_BUDGET`. On the 2026-01-25 slate, game `22500657` carried
0.602 from the same-conference model and 0.505 from the all-games model, and
both became order plans.

## The comparison

A rolling-season backtest (`make tune-decay`, since removed — see below), eight
folds. For each
evaluation season, train on everything up to three seasons earlier, validate on
the two in between, predict that season alone — the same regime the deployed
model runs in. Everything but the architecture is held fixed: xgboost only,
config hyperparameters, no random search, no calibration, and each arm's feature
list pinned to what its own Boruta run selected.

* **Split arm**: each game predicted by the same- or cross-conference model.
* **Unified arm**: every game predicted by the all-games model.

Both arms cover the identical 8,762 games (5,422 same-conference, 3,340 cross),
so the comparison is paired and the bootstrap resamples games.

Evaluation seasons 2014/15 – 2021/22, all strictly before the 2022/23 test
window. The test set was not touched.

| | log loss | Brier | accuracy |
|---|---|---|---|
| Split (routed) | 0.6265 | 0.2175 | 0.6533 |
| **Unified (one model)** | **0.6219** | **0.2159** | **0.6597** |

Paired difference, split − unified, in log loss: **+0.0047**, 95% CI
**[+0.0011, +0.0080]**. The interval excludes zero: the split is not merely
no better, it is **worse**.

Broken out by matchup, the loss is where the data is thinnest:

| matchup | games | Δ log loss (split − unified) | 95% CI |
|---|---|---|---|
| same conference | 5,422 | +0.0026 | [−0.0011, +0.0060] |
| cross conference | 3,340 | +0.0081 | [+0.0013, +0.0151] |

The same-conference model is a wash. The cross-conference model, which trains on
roughly a third of the rows, is what drags the split down — it pays more for the
data it gives up than it earns from specialising.

The reason is data volume, not a clever feature.

> **Correction, 2026-08-12.** This section used to claim that the split lost
> because `conference_diff_home_advantage_pct` already carries the interaction
> in the unified model's feature vector. That is false, and was false when
> written: Boruta-SHAP **rejects** that feature. It is absent from the 14
> selected features of the run this document was written against
> (`bd2afe2b`) and from the 17 of its successor (`63d60d95`) — in both it
> appears in `rejected`, not `selected`. The unified model therefore wins on
> rows alone: it trains on 2.6x the data the cross-conference model gets, and
> pays nothing for a conference interaction it never learns.

## What changed

* One deployed model. `configs/predict/predict_classifier.yaml` holds a single
  `model_uri`; the three-way routing loop, the three-URI guard and the
  `PredictionConfig.model_uris` map are gone.
* One row per game — `PREDICTION_KEY` in `src/ml/prediction/pipeline.py` is
  `["gameId"]`. Re-predicting a slate also clears the second row the old routing
  left behind.
* One config per *trained model*, not per conference: `configs/train/xgboost.yaml`
  is the deployed one. `train_same.yaml`, `train_different.yaml`,
  `train_all.yaml`, `xgb_same.yaml` and `xgb_different.yaml` are deleted, and
  `make train-all` is gone. (`all_models.yaml` was added later — it sweeps model
  *families* over the identical games, which is a different axis entirely from
  the conference routing removed here.)
* `src/betting/bets.py` raises if a date carries two rows for one game, so a
  duplicate is loud rather than expensive.
* One prediction key, defined once. `src/monitoring/scoring.py` imports
  `PREDICTION_KEY` rather than declaring its own. While the writer deduped on
  `gameId` and the scorer separately used `(gameId, conference_filter)`, the
  duplicate warning could not see the doubled rows at all: every pair differed
  in the second key. `prediction_scorecard.csv` reported 26 games for a 13-game
  slate, averaging two models' probabilities for one event into Brier and ECE.
* No `conference_filter` column on predictions, no `filters.conference_filter`
  in the config schema, no row masks in `build_splits` or the serving path, and
  MLflow artifacts named `nba_{model}` rather than `nba_{model}_{filter}`.

## The signal stayed; only the split went

`filters.conference_filter` and the three branches of
`apply_conference_features` survived this measurement for a while, so the
comparison would stay re-runnable. They are now gone too — the scaffolding cost
more to carry than a re-run is worth, and re-running it means checking out the
commit before their removal. The backtest harness itself
(`src/ml/training/backtest.py`, `tuning.py`, `src/cli/tune_season_decay.py`) went
the same way once cross-season decay was removed and it had no sweep axis left;
re-running any of this means rebuilding it.

What that removal did **not** touch is the conference signal itself. Two
features carry it, both built by `create_conference_features` for every game:

| feature | what it measures | same-conference |
|---|---|---|
| `conference_diff_home_advantage_pct` | the venue-balanced East/West gap, signed by who is hosting | 0.0 |
| `conference_home_court_advantage_pct` | how well the home team's conference holds its own floor against the other one, centered on parity | 0.0 |

The second one used to be reachable only from the `different` branch, under the
name `home_conference_vs_away_conference_record` and on a raw 0–1 scale. It is
kept because it is the complement of the first, not a duplicate:
`east_record_adjusted` averages a conference's hosting and visiting win rates,
so it discards venue on purpose, and this is exactly the discarded part.

Generalizing it to every game required a decision the split had let the codebase
avoid. Its ingredients are conference-season-level, defined on every date, so
the raw lookup returns a number for an all-East game — reporting "the East holds
home court at .550" on a game with no West team in it, and handing every East
home team one value and every West home team another. That is a confound, not
noise. Both features are therefore 0.0 within a conference, which is not an
imputation guess: when both teams are drawn from the same pool the conference
effect cancels exactly.

### Neither feature currently reaches the model

Boruta-SHAP rejects both. Run `63d60d95`, the first with
`conference_home_court_advantage_pct` available, selected 17 features and put
*both* conference columns in `rejected`. So the conference signal is correctly
computed, correctly defined for every game, and empirically not worth a column.

That is the measurement, not a bug: it is the same selection step every other
feature faces, and the answer for these two is currently no. Two consequences
worth keeping in view:

* Nothing downstream of feature selection depends on them, so the ETL work that
  builds the east/west tables is, today, computed for a feature that gets
  dropped. It is cheap, and it is what makes re-testing possible.
* The rejection is measured on a training window ending in 2019/20. If
  conference imbalance widens, the way to find out is to re-run selection, not
  to re-argue the case.

## Caveats

The eight folds evaluate 2014/15 – 2021/22. If conference imbalance widens
enough to change this, the way to find out is to re-run the comparison, not to
reintroduce the split on the strength of the original argument. Two of the folds
(2019/20, 2020/21) are COVID-affected. Per-fold results were written to
`outputs/tuning/`, which is gitignored, and the harness that produced them has
since been removed — the numbers above are the record, not a cache.
