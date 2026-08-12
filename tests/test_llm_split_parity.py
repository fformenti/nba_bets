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
            "pts_diff": rng.integers(-25, 25, N_GAMES),
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
            "games_played_HT": rng.integers(20, 60, N_GAMES),
            "games_played_VT": rng.integers(20, 60, N_GAMES),
            "record_L82_HT": rng.random(N_GAMES),
            "record_L82_VT": rng.random(N_GAMES),
            "pts_diff_avg_L82_HT": rng.normal(0, 5, N_GAMES),
            "pts_diff_avg_L82_VT": rng.normal(0, 5, N_GAMES),
        }
    )

    path = tmp_path / "games_features.csv"
    df.to_csv(path, index=False)

    return path, df


@pytest.fixture
def experiment_config(feature_table):
    path, _ = feature_table
    return ExperimentConfig(
        data={"path": str(path), "target_column": "winner", "date_column": "gameDate"},
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
            "intermediate_columns": ["pts_diff"],
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
    expected = source.set_index("gameId")["pts_diff"].to_dict()

    dataset = build_llm_dataset(llm_config, experiment_config)
    for game_id, completion in zip(
        dataset["test"]["game_id"], dataset["test"]["completion"], strict=True
    ):
        assert int(completion) == expected[game_id]


def test_prompts_do_not_contain_the_completion(experiment_config, llm_config):
    """Serialization is the leak boundary; assert it holds on real split output.

    Matches whole markdown cells rather than substrings: ``pts_diff_avg_L82_HT``
    is a legitimate feature that contains the leaking name ``pts_diff``.
    """
    dataset = build_llm_dataset(llm_config, experiment_config)

    for row in dataset["test"]:
        prompt_body = row["prompt"].rsplit("point differential of", 1)[0]
        assert "| pts_diff |" not in prompt_body
        assert "| winner |" not in prompt_body
        assert "| homeScore |" not in prompt_body
