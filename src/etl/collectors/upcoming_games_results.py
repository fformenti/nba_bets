"""Enrich upcoming game JSON files with final score metadata."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import requests
from nba_api.stats.endpoints import boxscoresummaryv3, boxscoresummaryv2

# NBA stats.nba.com is notoriously slow/unreliable.
# to avoid rate limiting. See: https://github.com/swar/nba_api/issues/320

from src.config.paths import (
    PROJECT_ROOT,
    UPCOMING_GAMES_DIR,
    UPCOMING_GAMES_RESULTS_DIR,
)
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")


NBA_API_TIMEOUT = 60
NBA_API_DELAY_SECONDS = 4
NBA_API_MAX_RETRIES = 2


@dataclass
class GameResult:
    home_score: int = 0
    away_score: int = 0
    overtimes: int = 0
    winner: int = 0
    postponed: int = 0
    attendance: int | None = None
    inactive_players: list[dict[str, Any]] = field(default_factory=list)


def _result_sets_by_name(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result_sets = payload.get("resultSets", [])
    mapped: dict[str, list[dict[str, Any]]] = {}
    for result_set in result_sets:
        name = result_set.get("name", "")
        headers = result_set.get("headers", [])
        rows = result_set.get("rowSet", [])
        mapped[name] = [dict(zip(headers, row)) for row in rows]
    return mapped


def _dataset_to_dicts(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    headers = dataset.get("headers", [])
    rows = dataset.get("data", [])
    return [dict(zip(headers, row)) for row in rows]


def _parse_game_status_overtimes(status_text: str | None) -> int:
    if not status_text:
        return 0
    match = re.search(r"(\d+)\s*OT", status_text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if re.search(r"\bOT\b", status_text, flags=re.IGNORECASE):
        return 1
    return 0


def _count_overtimes_from_line_score(line_score: dict[str, Any]) -> int | None:
    ot_columns = [
        column for column in line_score.keys() if column.upper().startswith("PTS_OT")
    ]
    if not ot_columns:
        return None
    overtime_count = 0
    for column in sorted(ot_columns):
        value = line_score.get(column)
        if value is not None and str(value) != "":
            overtime_count += 1
    return overtime_count


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _normalize_inactive_players(
    rows: list[dict[str, Any]], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    inactive_players = []
    for row in rows:
        normalized = {}
        for source_key, target_key in mapping.items():
            normalized[target_key] = row.get(source_key)
        inactive_players.append(normalized)
    return inactive_players


def _fetch_game_result_legacy(game_id: str) -> dict[str, Any]:
    try:
        response = boxscoresummaryv2.BoxScoreSummaryV2(
            game_id=game_id, timeout=NBA_API_TIMEOUT
        )
    except requests.Timeout:
        logger.warning("Timeout fetching game result for %s", game_id)
        return {"GameResult": GameResult(), "status": "timeout", "status_code": 500}

    payload = response.get_dict()
    result_sets = _result_sets_by_name(payload)

    summary_rows = result_sets.get("GameSummary", [])
    summary = summary_rows[0] if summary_rows else {}
    home_team_id = summary.get("HOME_TEAM_ID")
    away_team_id = summary.get("VISITOR_TEAM_ID")

    line_scores = result_sets.get("LineScore", [])
    home_line = None
    away_line = None
    for line in line_scores:
        team_id = line.get("TEAM_ID")
        if team_id == home_team_id:
            home_line = line
        elif team_id == away_team_id:
            away_line = line

    home_score = int(home_line) if home_line.get("PTS") is not None else 0
    away_score = int(away_line) if away_line.get("PTS") is not None else 0

    winner = 0
    overtime = 0
    postponed = 1
    inactive_players = []
    attendance = None
    if home_score > 0 and away_score > 0:
        winner = home_team_id if home_score > away_score else away_team_id
        postponed = 0

        overtime = _parse_game_status_overtimes(summary.get("GAME_STATUS_TEXT"))
        if overtime is None and home_line:
            overtime = _count_overtimes_from_line_score(home_line)

        game_info_rows = result_sets.get("GameInfo", [])
        game_info = game_info_rows[0] if game_info_rows else {}
        attendance = _coerce_int(
            game_info.get("ATTENDANCE") if game_info else summary.get("ATTENDANCE")
        )

        inactive_players = _normalize_inactive_players(
            result_sets.get("InactivePlayers", []),
            {
                "TEAM_ID": "teamId",
                "PLAYER_ID": "personId",
                "FIRST_NAME": "firstName",
                "LAST_NAME": "familyName",
                "JERSEY_NUM": "jerseyNum",
            },
        )

    return {
        "GameResult": GameResult(
            home_score=home_score,
            away_score=away_score,
            overtimes=overtime,
            winner=winner,
            postponed=postponed,
            attendance=attendance,
            inactive_players=inactive_players,
        ),
        "status": "success",
        "status_code": 200,
    }


def _fetch_game_result_v3(game_id: str) -> dict[str, Any]:
    try:
        response = boxscoresummaryv3.BoxScoreSummaryV3(
            game_id=game_id, timeout=NBA_API_TIMEOUT
        )
    except requests.Timeout:
        logger.warning("Timeout fetching game result for %s", game_id)
        return {"GameResult": GameResult(), "status": "timeout", "status_code": 500}

    summary_rows = _dataset_to_dicts(response.game_summary.get_dict())
    summary = summary_rows[0] if summary_rows else {}
    home_team_id = summary.get("homeTeamId")
    away_team_id = summary.get("awayTeamId")

    line_scores = _dataset_to_dicts(response.line_score.get_dict())
    home_line = None
    away_line = None
    for line in line_scores:
        team_id = line.get("teamId")
        if team_id == home_team_id:
            home_line = line
        elif team_id == away_team_id:
            away_line = line

    home_score = (
        int(home_line.get("score")) if home_line.get("score") is not None else 0
    )
    away_score = (
        int(away_line.get("score")) if away_line.get("score") is not None else 0
    )

    winner = 0
    inactive_players = []
    attendance = None
    overtime = 0
    postponed = 1
    if home_score > 0 and away_score > 0:
        postponed = 0
        winner = home_team_id if home_score > away_score else away_team_id
        overtime = _parse_game_status_overtimes(summary.get("gameStatusText"))
        if overtime > 0:
            period = summary.get("period")
            try:
                period_value = int(period) if period is not None else None
            except (TypeError, ValueError):
                period_value = None
            if period_value > 4:
                overtime = period_value - 4

        game_info_rows = _dataset_to_dicts(response.game_info.get_dict())
        game_info = game_info_rows[0] if game_info_rows else {}
        attendance = _coerce_int(game_info.get("attendance"))

        inactive_rows = _dataset_to_dicts(response.inactive_players.get_dict())
        inactive_players = _normalize_inactive_players(
            inactive_rows,
            {
                "teamId": "teamId",
                "personId": "personId",
                "firstName": "firstName",
                "familyName": "familyName",
            },
        )

    return {
        "GameResult": GameResult(
            home_score=home_score,
            away_score=away_score,
            overtimes=overtime,
            winner=winner,
            postponed=postponed,
            attendance=attendance,
            inactive_players=inactive_players,
        ),
        "status": "success",
        "status_code": 200,
    }


def _fetch_game_result(game_id: str) -> dict[str, Any]:
    """Fetch game result, trying v3 first, then legacy on timeout. Returns dict with GameResult and status."""
    for attempt in range(NBA_API_MAX_RETRIES + 1):
        try:
            result = _fetch_game_result_v3(game_id)
            if result["status"] == "timeout":
                logger.info("Retrying with legacy endpoint for %s", game_id)
                time.sleep(NBA_API_DELAY_SECONDS)
                result = _fetch_game_result_legacy(game_id)
            if result["status"] != "timeout":
                return result
            if attempt < NBA_API_MAX_RETRIES:
                logger.info(
                    "Retrying game %s (attempt %d/%d)",
                    game_id,
                    attempt + 2,
                    NBA_API_MAX_RETRIES + 1,
                )
                time.sleep(NBA_API_DELAY_SECONDS)
        except Exception as exc:
            logger.warning(
                "Using BoxScoreSummary Legacy version for %s: %s", game_id, exc
            )
            return _fetch_game_result_legacy(game_id)
    return {"GameResult": GameResult(), "status": "timeout", "status_code": 500}


def enrich_upcoming_game_result(input_path: Path, output_dir: Path) -> Path:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_game_id = payload.get("gameId")
    if raw_game_id is None:
        raise ValueError(f"Missing gameId in {input_path}")
    # add leading zeros to make it 10 digits
    game_id = f"{raw_game_id:010d}"

    logger.info("Fetching final score for game %s", game_id)
    result_dict = _fetch_game_result(game_id)

    if result_dict["status"] == "timeout":
        return

    game_info = result_dict["GameResult"]
    payload["homeTeamFinalScore"] = game_info.home_score
    payload["awayTeamFinalScore"] = game_info.away_score
    payload["overtimes"] = game_info.overtimes
    payload["winner"] = game_info.winner
    payload["postponed"] = game_info.postponed
    payload["attendance"] = game_info.attendance
    payload["inactivePlayers"] = game_info.inactive_players

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    # delete JSON file from input directory after saving on output directory
    input_path.unlink()

    # Remove PROJECT_ROOT from output_path name
    output_short_name = output_path.relative_to(PROJECT_ROOT)
    logger.info("Saved enriched game to %s", output_short_name)
    return


def enrich_upcoming_games_results(
    input_dir: Path,
    output_dir: Path,
) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, input_path in enumerate(sorted(input_dir.glob("*.json"))):
        if i > 0:
            time.sleep(NBA_API_DELAY_SECONDS)
        enrich_upcoming_game_result(input_path, output_dir)
    return


def main() -> None:
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(
        description="Enrich upcoming game JSON files with final scores."
    )
    parser.add_argument("input_path", type=Path, nargs="?")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=UPCOMING_GAMES_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UPCOMING_GAMES_RESULTS_DIR,
    )
    args = parser.parse_args()

    enrich_upcoming_games_results(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
