"""Unreferenced functions removed from ``src/`` during the 2026-08 refactor.

Nothing imports this module and it is excluded from ruff and pytest. It exists
so the code is findable without a git archaeology session: each function below
carries the path and line it came from at commit 78b5836.

A function landed here if a whole-repo search (``src/``, ``tests/``,
``configs/``, ``Makefile``, notebooks) found no reference other than its own
definition and its re-export in a package ``__init__``. To revive one, move it
back to the original module and restore the ``__all__`` entry.

Imports are deliberately omitted — this file is a record, not a library.
"""

# ---------------------------------------------------------------------------
# from src/etl/features/teams_arena.py:97
# Nothing loaded the arena table back from disk; build_teams_arena() is called
# in-process by make_features.py and the DataFrame is passed straight on.
# Needs: ast, pandas as pd
# ---------------------------------------------------------------------------


def load_teams_arena(path) -> "pd.DataFrame":  # noqa: F821
    """Load teams_arena CSV, parsing the ``home_arena_ids`` string column back to list."""
    df = pd.read_csv(path)  # noqa: F821
    df["home_arena_ids"] = df["home_arena_ids"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v  # noqa: F821
    )
    return df


# ---------------------------------------------------------------------------
# from src/etl/collectors/fetch_game/get_teams_locations.py:41
# The agent in that module returns NBALocation; the team_name-carrying subclass
# was never instantiated.
# Needs: pydantic.Field, NBALocation
# ---------------------------------------------------------------------------


class NBATeam(NBALocation):  # noqa: F821
    """Class to hold a NBA team name and its city and state."""

    team_name: str = Field(description="The name of the NBA team")  # noqa: F821


# ---------------------------------------------------------------------------
# from src/ml/config/loader.py:201
# Superseded by the pydantic schemas in src/ml/config/schema.py, which validate
# on construction. No caller ever invoked this.
# Needs: typing.Dict, typing.Any, get_nested_config (still in loader.py)
# ---------------------------------------------------------------------------


def validate_config(config: "Dict[str, Any]", required_keys: list[str]) -> None:  # noqa: F821
    """
    Validate that required configuration keys are present.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    required_keys : list
        List of required key paths (dot notation)

    Raises
    ------
    ValueError
        If any required keys are missing
    """
    missing_keys = []

    for key_path in required_keys:
        if get_nested_config(config, key_path) is None:  # noqa: F821
            missing_keys.append(key_path)

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


# ---------------------------------------------------------------------------
# from src/ml/prediction/io.py:68
# The prediction pipeline loads historical features through
# src/ml/training/data_prep.py::load_and_validate_data instead, which also
# validates required columns and row count.
# Needs: pathlib.Path, pandas as pd
# ---------------------------------------------------------------------------


def load_historical_features(features_path: "Path") -> "pd.DataFrame":  # noqa: F821
    """
    Load historical features used to build inference rows.
    """
    features_path = Path(features_path)  # noqa: F821
    if not features_path.exists():
        raise FileNotFoundError(f"Historical features not found: {features_path}")

    df = pd.read_csv(features_path, parse_dates=["gameDate"])  # noqa: F821
    if df.empty:
        raise ValueError(f"Historical features file is empty: {features_path}")
    return df


# ---------------------------------------------------------------------------
# from src/ml/scripts/place_bets.py:69
# The flat-stake sizing alternative. buy_shares() only ever called the linear
# (edge-proportional) variant.
# ---------------------------------------------------------------------------


def find_shares_to_buy_fixed_investment(ask_price, budget):
    investment = budget
    shares_to_buy = int(investment / ask_price)
    return shares_to_buy


# ---------------------------------------------------------------------------
# from src/ml/training/utils.py:459 and :463
# Thin wrappers over the tester classes. Callers construct
# RegressionTester/ClassificationTester directly and call .run().
# Needs: DEFAULT_SIZE, RegressionTester, ClassificationTester
#        (renamed from Tester_Regressors / Tester_Classifiers)
# ---------------------------------------------------------------------------


def evaluate(function, data, size=DEFAULT_SIZE):  # noqa: F821
    Tester_Regressors(function, data, size=size).run()  # noqa: F821


def evaluate_classifier(function, data, size=DEFAULT_SIZE):  # noqa: F821
    Tester_Classifiers(function, data, size=size).run()  # noqa: F821


# ---------------------------------------------------------------------------
# from src/ml/models/trainer.py:365 and :404
# Convenience wrappers over ModelTrainer. src/ml/training/runners.py drives the
# ModelTrainer directly because it needs the trainer object for CV and MLflow.
# Needs: BaseEstimator, pandas as pd, typing.Literal/Optional/Dict/Any,
#        ModelTrainer
# ---------------------------------------------------------------------------


def train_model(
    model: "BaseEstimator",  # noqa: F821
    X_train: "pd.DataFrame",  # noqa: F821
    y_train: "pd.Series",  # noqa: F821
    task_type: "Literal['regression', 'classification']",  # noqa: F821
    X_val: "Optional[pd.DataFrame]" = None,  # noqa: F821
    y_val: "Optional[pd.Series]" = None,  # noqa: F821
    random_state: "Optional[int]" = None,  # noqa: F821
) -> "tuple[BaseEstimator, Dict[str, Any]]":  # noqa: F821
    """
    Convenience function to train a model.

    Returns
    -------
    tuple
        Trained model and training metrics
    """
    trainer = ModelTrainer(model, task_type, random_state)  # noqa: F821
    metrics = trainer.train(X_train, y_train, X_val, y_val)
    return trainer.model, metrics


def evaluate_model(
    model: "BaseEstimator",  # noqa: F821
    X: "pd.DataFrame",  # noqa: F821
    y: "pd.Series",  # noqa: F821
    task_type: "Literal['regression', 'classification']",  # noqa: F821
    prefix: str = "test",
) -> "Dict[str, float]":  # noqa: F821
    """
    Convenience function to evaluate a model.

    Returns
    -------
    dict
        Evaluation metrics
    """
    trainer = ModelTrainer(model, task_type)  # noqa: F821
    trainer.is_fitted = True
    trainer.model = model
    return trainer.evaluate(X, y, prefix)


# ---------------------------------------------------------------------------
# from src/ml/models/registry.py:143
# Models are persisted through MLflow (src/ml/tracking/mlflow_tracker.py), not
# to the local models/ directory. ModelRegistry.save() covers the local path.
# Needs: pathlib.Path, joblib, json, BaseEstimator, typing.Optional/Dict/Any
# ---------------------------------------------------------------------------


def save_model(
    model: "BaseEstimator",  # noqa: F821
    file_path: "Path",  # noqa: F821
    metadata: "Optional[Dict[str, Any]]" = None,  # noqa: F821
) -> "Path":  # noqa: F821
    """
    Save a model to disk.

    Returns
    -------
    Path
        Path to saved model file
    """
    file_path = Path(file_path)  # noqa: F821
    file_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, file_path)  # noqa: F821

    if metadata:
        metadata_path = file_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)  # noqa: F821

    return file_path


# ---------------------------------------------------------------------------
# from src/ml/datasets/splitters.py:240
# Every experiment splits temporally or on the frozen holdout — stratifying
# would leak future games into training.
# Needs: pandas as pd, typing.Optional/Tuple, train_val_test_split
# ---------------------------------------------------------------------------


def stratified_split(
    X: "pd.DataFrame",  # noqa: F821
    y: "pd.Series",  # noqa: F821
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: "Optional[int]" = None,  # noqa: F821
) -> "Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]":  # noqa: F821
    """
    Split data with stratification to maintain class distribution.

    This is a convenience wrapper around train_val_test_split with stratify=y.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    return train_val_test_split(  # noqa: F821
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# from src/ml/evaluation/visualization.py:225
# Nothing computes learning curves; the experiment logs CV results instead.
# Needs: numpy as np, matplotlib.pyplot as plt
# ---------------------------------------------------------------------------


def plot_learning_curves(
    train_scores: "np.ndarray",  # noqa: F821
    val_scores: "np.ndarray",  # noqa: F821
    train_sizes: "np.ndarray",  # noqa: F821
    title: str = "Learning Curves",
    figsize: tuple = (10, 6),
) -> "plt.Figure":  # noqa: F821
    """
    Plot learning curves.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)  # noqa: F821

    train_mean = np.mean(train_scores, axis=1)  # noqa: F821
    train_std = np.std(train_scores, axis=1)  # noqa: F821
    val_mean = np.mean(val_scores, axis=1)  # noqa: F821
    val_std = np.std(val_scores, axis=1)  # noqa: F821

    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training Score")
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )

    ax.plot(train_sizes, val_mean, "o-", color="red", label="Validation Score")
    ax.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red"
    )

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()  # noqa: F821

    return fig


# ---------------------------------------------------------------------------
# from src/ml/training/data_prep.py:12
# Its only reference was an unused import in experiment.py:48. The minimum
# games-played filter is applied via the feature config instead.
# Needs: pandas as pd
# ---------------------------------------------------------------------------


def filter_minimum_games_played(
    df: "pd.DataFrame", minimum_games: int = 15
) -> "pd.DataFrame":  # noqa: F821
    return df[
        (df["games_played_HT"] > minimum_games)
        & (df["games_played_VT"] > minimum_games)
    ]
