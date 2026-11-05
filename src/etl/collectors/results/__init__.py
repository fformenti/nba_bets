"""Pluggable sources of finished-game outcomes.

Register a new provider by adding it to :data:`SOURCES`; callers select one by
name and never import the implementation directly.
"""

from __future__ import annotations

from src.etl.collectors.results.base import GameResult, ResultsSource
from src.etl.collectors.results.nba_api_source import NBAApiResultsSource
from src.etl.collectors.results.placeholder_source import PlaceholderResultsSource
from src.etl.collectors.results.release_source import ReleaseResultsSource

SOURCES = {
    NBAApiResultsSource.name: NBAApiResultsSource,
    PlaceholderResultsSource.name: PlaceholderResultsSource,
    ReleaseResultsSource.name: ReleaseResultsSource,
}

DEFAULT_SOURCE = NBAApiResultsSource.name


def get_results_source(name: str = DEFAULT_SOURCE, **kwargs) -> ResultsSource:
    """Instantiate a results source by name.

    Parameters
    ----------
    name : str
        One of :data:`SOURCES`. ``'nba_api'`` is stats.nba.com; ``'release'``
        reads the sportsdataverse GitHub releases, which also give the date a
        game was played; ``'placeholder'`` reads hand-dropped files and exists
        so the pipeline can be run end to end without a live feed.
    **kwargs
        Passed to the source's constructor.
    """
    if name not in SOURCES:
        raise ValueError(
            f"Unknown results source '{name}'. Available: {sorted(SOURCES)}."
        )
    return SOURCES[name](**kwargs)


__all__ = [
    "DEFAULT_SOURCE",
    "SOURCES",
    "GameResult",
    "NBAApiResultsSource",
    "PlaceholderResultsSource",
    "ReleaseResultsSource",
    "ResultsSource",
    "get_results_source",
]
