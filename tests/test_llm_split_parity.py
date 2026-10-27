"""The LLM and the sklearn models must be scored on the same games.

If the two families train or test on different gameIds, "did the LLM beat the
ML models?" is unanswerable — which was the state before the LLM dataset was
rebuilt on top of ``build_splits``. These tests pin the parity.

They run on a synthetic feature table written to a tmp_path, so no real data or
network access is required.
"""

import numpy as np
import pandas as pd
import pytest

from src.config.constants import RESPONSE_VARIABLES
from src.ml.config.schema import (
    ExperimentConfig,
    FeatureGroupConfig,
    FeaturesMapConfig,
    LLMTrainingConfig,
)
from src.ml.datasets.splits import build_splits
from src.ml.llm.dataset import build_llm_dataset

# Four seasons: one trains, two validate, one tests. The season-based splitter
# needs at least val_seasons + 1 seasons before the test boundary.
SEASONS = ("2019/20", "2020/21", "2021/22", "2022/23")
TEST_START_SEASON = "2022/23"
VAL_SEASONS = 2
GAMES_PER_SEASON = 100
N_GAMES = GAMES_PER_SEASON * len(SEASONS)
SPLITS = ("train", "validation", "test")


def _record_only(lags: list[int]) -> FeaturesMapConfig:
    """Only the record group enabled, so the synthetic table stays small.

    Built from the model fields so groups added later default to off rather
    than demanding columns this fixture does not have.
    """
    groups = {
        name: FeatureGroupConfig(enabled=False, lags=[])
        for name in FeaturesMapConfig.model_fields
    }
    groups["record"] = FeatureGroupConfig(lags=lags, delta=True, enabled=True)
    return FeaturesMapConfig(**groups)


@pytest.fixture
def feature_table(tmp_path):
    """A minimal but structurally faithful games_features.csv."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2019-10-01", periods=N_GAMES, freq="D")

    df = pd.DataFrame(
        {
            "gameId": np.arange(21900001, 21900001 + N_GAMES),
            "gameDate": dates,
            "gameDateOnlyStr": dates.strftime("%Y-%m-%d"),
            "season": [s for s in SEASONS for _ in range(GAMES_PER_SEASON)],
            "winner": rng.integers(0, 2, N_GAMES),
            "win_bool": rng.integers(0, 2, N_GAMES),
            "margin": rng.integers(-25, 25, N_GAMES),
            "margin_scaled": rng.normal(0, 1, N_GAMES),
            "hometeamId": rng.integers(1, 6, N_GAMES),
            "awayteamId": rng.integers(6, 11, N_GAMES),
            "hometeamName": "Home",
            "awayteamName": "Away",
            "hometeamConference": "East",
            "awayteamConference": "East",
            # The conference features are built for every game now, so the
            # synthetic table has to carry their east/west inputs. Every fixture
            # row is East-vs-East, which makes both features 0.0.
            "east_record_adjusted": 0.55,
            "west_record_adjusted": 0.45,
            "east_record_at_east": 0.60,
            "west_record_at_west": 0.52,
            # Well above saturation_games (20), so the ramp sits at full weight
            # on the current-season columns above regardless of what
            # last_season_* holds; real (non-NaN) values here since
            # prepare_data's drop_na=True would otherwise drop every row.
            "games_played_east_vs_west": 999,
            "last_season_east_record_adjusted": 0.55,
            "last_season_west_record_adjusted": 0.45,
            "last_season_east_record_at_east": 0.60,
            "last_season_west_record_at_west": 0.52,
            "games_played_HT": rng.integers(20, 60, N_GAMES),
            "games_played_VT": rng.integers(20, 60, N_GAMES),
            "record_L82_HT": rng.random(N_GAMES),
            "record_L82_VT": rng.random(N_GAMES),
            "margin_avg_L82_HT": rng.normal(0, 5, N_GAMES),
            "margin_avg_L82_VT": rng.normal(0, 5, N_GAMES),
        }
    )

    path = tmp_path / "games_features.csv"
    df.to_csv(path, index=False)

    return path, df


@pytest.fixture
def experiment_config(feature_table):
    path, _ = feature_table
    return ExperimentConfig(
        data={"path": str(path), "target_column": "win_bool", "date_column": "gameDate"},
        filters={
            "minimum_games_train": 0,
            "minimum_games_test": 0,
            "min_season": None,
        },
        feature_engineering={
            "selection_mode": "exclusion",
            "features": _record_only(lags=[82]),
            "metadata_columns": [
                "gameId",
                "season",
                "gameDateOnlyStr",
                "hometeamName",
                "awayteamName",
                "hometeamConference",
                "awayteamConference",
                "games_played_HT",
                "games_played_VT",
            ],
            "intermediate_columns": [],
        },
        splitting={
            "test_start_season": TEST_START_SEASON,
            "val_seasons": VAL_SEASONS,
        },
    )


@pytest.fixture
def llm_config():
    return LLMTrainingConfig(data={"serialization_format": "markdown"})


def test_game_ids_match_per_split(experiment_config, llm_config):
    """Set equality, not subsetting — a shifted split would still subset."""
    splits = build_splits(experiment_config)
    dataset = build_llm_dataset(llm_config, experiment_config)

    for name in SPLITS:
        llm_ids = {int(g) for g in dataset[name]["game_id"]}
        ml_ids = set(splits.game_ids(name).astype(int))
        assert llm_ids == ml_ids, f"{name} split diverged"


@pytest.mark.parametrize("target", RESPONSE_VARIABLES)
def test_no_response_variable_reaches_the_features(experiment_config, target):
    """Whichever outcome is the target, the other two restate it."""
    config = experiment_config.model_copy(
        update={"data": experiment_config.data.model_copy(update={"target_column": target})}
    )
    splits = build_splits(config)

    assert splits.y_train.name == target
    for X in (splits.X_train, splits.X_val, splits.X_test):
        assert not set(RESPONSE_VARIABLES) & set(X.columns)


def test_splits_are_pairwise_disjoint(experiment_config, llm_config):
    dataset = build_llm_dataset(llm_config, experiment_config)
    train, val, test = ({int(g) for g in dataset[s]["game_id"]} for s in SPLITS)

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_every_game_is_serialized_once(experiment_config, llm_config):
    dataset = build_llm_dataset(llm_config, experiment_config)

    for name in SPLITS:
        ids = [int(g) for g in dataset[name]["game_id"]]
        assert len(ids) == len(set(ids)), f"{name} has duplicate gameIds"
        assert len(dataset[name]["prompt"]) == len(ids)


def test_completion_is_the_signed_point_differential(
    experiment_config, llm_config, feature_table
):
    """The completion must be the real margin for that game, sign included."""
    _, source = feature_table
    expected = source.set_index("gameId")["margin"].to_dict()

    dataset = build_llm_dataset(llm_config, experiment_config)
    for game_id, completion in zip(
        dataset["test"]["game_id"], dataset["test"]["completion"], strict=True
    ):
        assert int(completion) == expected[game_id]


def test_prompts_do_not_contain_the_completion(experiment_config, llm_config):
    """Serialization is the leak boundary; assert it holds on real split output.

    Matches whole markdown cells rather than substrings: ``margin_avg_L82_HT``
    is a legitimate feature that contains the leaking name ``margin``.
    """
    dataset = build_llm_dataset(llm_config, experiment_config)

    for row in dataset["test"]:
        prompt_body = row["prompt"].rsplit("point differential of", 1)[0]
        assert "| margin |" not in prompt_body
        assert "| winner |" not in prompt_body
        assert "| homeScore |" not in prompt_body


def _with_neutral_games(feature_table, experiment_config):
    """Rewrite the table with every 10th game neutral and turn the filter on."""
    path, df = feature_table
    df = df.assign(neutral_court=(np.arange(len(df)) % 10 == 0).astype(int))
    df.to_csv(path, index=False)
    config = experiment_config.model_copy(
        update={
            "filters": experiment_config.filters.model_copy(
                update={"exclude_neutral_court": True}
            )
        }
    )
    return config, set(df.loc[df["neutral_court"] == 1, "gameId"])


def test_neutral_court_games_are_excluded_from_every_split(
    feature_table, experiment_config, llm_config
):
    config, neutral_ids = _with_neutral_games(feature_table, experiment_config)

    splits = build_splits(config)
    dataset = build_llm_dataset(llm_config, config)

    for name in SPLITS:
        ml_ids = set(splits.game_ids(name).astype(int))
        llm_ids = {int(g) for g in dataset[name]["game_id"]}
        assert ml_ids, f"{name} split is empty"
        assert not ml_ids & neutral_ids, f"{name} kept a neutral game"
        assert llm_ids == ml_ids, f"{name} split diverged"


def test_neutral_court_filter_needs_the_column(experiment_config):
    config = experiment_config.model_copy(
        update={
            "filters": experiment_config.filters.model_copy(
                update={"exclude_neutral_court": True}
            )
        }
    )

    with pytest.raises(ValueError, match="neutral_court"):
        build_splits(config)


def _stub_per_team(monkeypatch):
    """per_team without its column set: one line per game, then the cue."""
    import src.ml.llm.dataset as dataset_module

    monkeypatch.setattr(dataset_module, "load_prompt_config", lambda path: {})
    monkeypatch.setattr(
        dataset_module,
        "serialize_frame",
        lambda features, metadata, fmt, raw, prompt_config, cue: [f"game {i}\n\n{cue}" for i in features.index],
    )


def test_both_target_modes_score_the_same_games(experiment_config, feature_table, monkeypatch):
    """Classification vs regression is one change: same games, same prompts but the cue."""
    _stub_per_team(monkeypatch)
    _, source = feature_table
    margins = source.set_index("gameId")["margin"].to_dict()
    built = {
        mode: build_llm_dataset(
            LLMTrainingConfig(
                data={"serialization_format": "per_team", "target_mode": mode, "dataset_name": f"x/nba-bets-{mode}"}
            ),
            experiment_config,
        )
        for mode in ("regression", "classification")
    }

    for name in SPLITS:
        regression, classification = built["regression"][name], built["classification"][name]
        assert regression["game_id"] == classification["game_id"]
        assert [p.removesuffix("Home margin =") for p in regression["prompt"]] == [
            p.removesuffix("Home team") for p in classification["prompt"]
        ]
        for game_id, completion in zip(classification["game_id"], classification["completion"], strict=True):
            assert completion == (" won" if margins[game_id] > 0 else " lost")
        # The real margin rides along in both, so truth never parses " won".
        assert classification["margin"] == regression["margin"]


def test_classification_needs_the_per_team_format(experiment_config):
    config = LLMTrainingConfig(
        data={"serialization_format": "markdown", "target_mode": "classification", "dataset_name": "x/d-classification"}
    )
    with pytest.raises(ValueError, match="per_team"):
        build_llm_dataset(config, experiment_config)
