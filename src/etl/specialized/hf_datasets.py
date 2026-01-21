"""HuggingFace dataset creation utilities."""

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login

# Add project root to path to allow imports when running directly
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
print(_project_root)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import LOCAL_GAMES_FEATURES_PATH
from src.etl.specialized.games_class import Game


def fix_float_to_int(df):
    """Fix float columns to int for rest days."""
    df["rested_days_HT"] = df["rested_days_HT"].astype("int")
    df["days_at_home"] = df["days_at_home"].astype("int")
    df["rested_days_VT"] = df["rested_days_VT"].astype("int")
    df["days_on_road"] = df["days_on_road"].astype("int")
    return df


def make_huggingface_dataset(list_of_games: list[Game], test=False) -> Dataset:
    """
    Create a HuggingFace dataset from a list of Game objects.

    Parameters
    ----------
    list_of_games : list[Game]
        List of Game objects
    test : bool, default=False
        Whether to use test prompts (without outcomes)

    Returns
    -------
    Dataset
        HuggingFace Dataset object
    """
    assert isinstance(test, bool)

    game_id: list[int] = [i.game_id for i in list_of_games]
    if test:
        prompts: list[str] = [i.get_test_prompt() for i in list_of_games]
    else:
        prompts: list[str] = [i.prompt for i in list_of_games]  # type: ignore
    point_diff: list[int] = [i.point_diff for i in list_of_games]

    return Dataset.from_dict(
        {
            "game_id": game_id,
            "text": prompts,
            "point_diff": point_diff,
        }
    )


def upload_to_huggingface(dataset):
    """
    Upload dataset to HuggingFace Hub.

    Parameters
    ----------
    dataset : DatasetDict
        HuggingFace dataset to upload
    """
    hf_token = os.getenv("HF_llm_training_token")
    login(hf_token, add_to_git_credential=True)

    HF_USER = "fformenti"
    DATASET_NAME = f"{HF_USER}/nba-bets"
    dataset.push_to_hub(DATASET_NAME, private=True)


def main():
    """Main function to create and upload HuggingFace dataset."""
    games: pd.DataFrame = pd.read_csv(LOCAL_GAMES_FEATURES_PATH)
    games_filtered_nulls: pd.DataFrame = games.dropna().copy(deep=True)
    games_filtered_fixed: pd.DataFrame = fix_float_to_int(games_filtered_nulls)
    games_dict = games_filtered_fixed.to_dict(orient="records")
    games_objs: list[Game] = [Game(i) for i in games_dict]

    train_end_idx: int = round(len(games_objs) * 0.80)
    validation_end_idx: int = round(len(games_objs) * 0.90)
    train: list[Game] = games_objs[:train_end_idx]
    validation: list[Game] = games_objs[train_end_idx:validation_end_idx]
    test: list[Game] = games_objs[validation_end_idx:]

    # Save pickle files
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    with open(data_dir / "train_prompt.pkl", "wb") as file:
        pickle.dump(train, file)

    with open(data_dir / "validation_prompt.pkl", "wb") as file:
        pickle.dump(validation, file)

    with open(data_dir / "test_prompt.pkl", "wb") as file:
        pickle.dump(test, file)

    train_dataset: Dataset = make_huggingface_dataset(train)
    validation_dataset: Dataset = make_huggingface_dataset(validation, test=True)
    test_dataset: Dataset = make_huggingface_dataset(test, test=True)

    dataset = DatasetDict(
        {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}
    )
    upload_to_huggingface(dataset)


if __name__ == "__main__":
    main()
