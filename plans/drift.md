 Plan: Redesign Sample Weighting Strategy

 Context

 The EDA report (W1) shows home win rate drifting from 60.1% (train, 1995-2014) to 55.0% (test, 2020-2026). The project currently has within-season weighting to
 downweight early-season games, but no cross-season weighting to handle temporal drift. Additionally, the within-season weighting is a complete no-op in
 train_same.yaml because minimum_games_train=30 already filters out all games that would receive reduced weight under saturation_K=30.

 Two independent concerns need separate strategies:
 1. Within-season: early games carry less signal (features haven't stabilized)
 2. Cross-season: older seasons may not reflect current league dynamics

 ---
 1. Within-Season Weighting (fix existing)

 Current formula (keep as-is): weight = clip(min(games_played_HT, games_played_VT) / K, 0.0, 1.0)

 Problem: In train_same.yaml, minimum_games_train=30 hard-filters all games where either team has played ≤ 30 games. With saturation_K=30, every surviving game gets
 weight 1.0. The weighting does nothing.

 Fix: Lower minimum_games_train to 15 in train_same.yaml (already 15 in the other two configs). This lets games from game 16-30 pass through with reduced weights
 (0.53–0.97), giving the model a smooth signal transition instead of a hard cutoff.

 Changes

 - configs/train/train_same.yaml line 17: change minimum_games_train: 30 → minimum_games_train: 15

 ---
 2. Cross-Season Weighting (new)

 Formula: cross_season_weight = exp(-λ × (max_season_idx - sample_season_idx))

 - season_idx: ordinal index of each season within the training set (0 = oldest, N = most recent)
 - max_season_idx: the highest season index in the training set
 - λ (lambda): decay rate hyperparameter (default 0.1)

 Examples with λ=0.1 across 20 training seasons:

 ┌─────────────────┬────────┐
 │   Seasons ago   │ Weight │
 ├─────────────────┼────────┤
 │ 0 (most recent) │ 1.00   │
 ├─────────────────┼────────┤
 │ 5               │ 0.61   │
 ├─────────────────┼────────┤
 │ 10              │ 0.37   │
 ├─────────────────┼────────┤
 │ 15              │ 0.22   │
 ├─────────────────┼────────┤
 │ 20              │ 0.14   │
 └─────────────────┴────────┘

 Why exponential decay:
 - Single tunable hyperparameter (λ)
 - Smooth, continuous — no arbitrary cutoffs
 - Well-motivated: more recent data is more representative, but old data still contributes
 - Standard approach for temporal weighting in non-stationary environments

 Final combined weight: final_weight = within_season_weight × cross_season_weight
 - Both independently in [0, 1], product also in [0, 1]
 - Multiplicative combination is natural since the concerns are independent

 ---
 Implementation Steps

 Step 1: Update config schema

 File: src/ml/config/schema.py — WeightingConfig class (lines 22-30)

 Add two fields:
 class WeightingConfig(BaseModel):
     model_config = ConfigDict(extra="ignore")

     enabled: bool = True
     saturation_K: int = Field(
         default=30,
         description="Games-played saturation point for sample weights. "
         "Observations where min(games_played_HT, games_played_VT) >= K get weight 1.0.",
     )
     season_decay_enabled: bool = Field(
         default=False,
         description="Enable exponential decay weighting by season to downweight older seasons.",
     )
     season_decay_lambda: float = Field(
         default=0.1,
         description="Decay rate for cross-season weighting. Higher = faster decay. "
         "Weight = exp(-lambda * seasons_ago).",
     )

 Step 2: Compute cross-season weights in experiment runner

 File: src/ml/training/experiment.py — after existing within-season weight computation (~line 235)

 Logic:
 1. Extract season column from metadata for training indices
 2. Map seasons to ordinal indices (sorted chronologically)
 3. Compute max_season_idx from training set only (no leakage)
 4. Compute cross_season_weight = exp(-λ × (max_idx - season_idx)) per sample
 5. Multiply with existing train_sample_weight
 6. Log cross-season weight stats (min, mean, pct_full_weight)

 if weighting_config.season_decay_enabled:
     lam = weighting_config.season_decay_lambda
     train_seasons = metadata.loc[X_train.index, "season"]
     season_order = sorted(train_seasons.unique())
     season_to_idx = {s: i for i, s in enumerate(season_order)}
     max_idx = len(season_order) - 1
     season_indices = train_seasons.map(season_to_idx).to_numpy(dtype=np.float64)
     cross_season_weight = np.exp(-lam * (max_idx - season_indices))
     train_sample_weight *= cross_season_weight
     logger.info(
         f"Season decay enabled (λ={lam}): "
         f"min_weight={cross_season_weight.min():.3f}, "
         f"mean_weight={cross_season_weight.mean():.3f}, "
         f"oldest_season={season_order[0]}, newest={season_order[-1]}"
     )

 Step 3: Update training configs

 Files: All three YAML configs

 configs/train/train_same.yaml:
 filters:
   minimum_games_train: 15    # was 30 (redundant with saturation_K)

 weighting:
   enabled: true
   saturation_K: 30
   season_decay_enabled: true
   season_decay_lambda: 0.1

 configs/train/train_different.yaml and configs/train/train_all.yaml:
 weighting:
   enabled: true
   saturation_K: 30
   season_decay_enabled: true
   season_decay_lambda: 0.1

 Step 4: Log combined weight to MLflow

 File: src/ml/training/experiment.py — in the params logging block (~line 102)

 Add:
 "season_decay_enabled": weighting_config.season_decay_enabled,
 "season_decay_lambda": weighting_config.season_decay_lambda,

 ---
 Files Modified (summary)

 ┌────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────┐
 │                File                │                                    Change                                     │
 ├────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ src/ml/config/schema.py            │ Add season_decay_enabled, season_decay_lambda to WeightingConfig              │
 ├────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ src/ml/training/experiment.py      │ Compute cross-season weights, multiply with within-season, log stats + params │
 ├────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ configs/train/train_same.yaml      │ Fix minimum_games_train: 15, add season decay config                          │
 ├────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ configs/train/train_different.yaml │ Add season decay config                                                       │
 ├────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ configs/train/train_all.yaml       │ Add season decay config                                                       │
 └────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────┘

 ---
 Verification

 1. Unit check: Run training with train_same.yaml and verify log output shows:
   - Within-season weights with values < 1.0 (confirms the min_games fix worked)
   - Cross-season weights with a range (e.g., min ~0.14, mean ~0.5 for λ=0.1)
 2. Backwards compat: season_decay_enabled: false (default) should produce identical results to current behavior
 3. No leakage: Season indices are derived only from training set seasons
 4. Compare runs: Train with and without season decay, compare val/test metrics in MLflow to measure impact