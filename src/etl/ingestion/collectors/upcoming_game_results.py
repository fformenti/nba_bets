"""Enrich upcoming game JSON files with final score metadata."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nba_api.stats.endpoints import boxscoresummaryv2, boxscoresummaryv3

from src.config import RAW_INCREMENTAL_DIR
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class GameResult:
    home_score: int | None
    away_score: int | None
    overtimes: int | None
    attendance: int | None
    inactive_players: list[dict[str, Any]]


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


def _parse_game_status_overtimes(status_text: str | None) -> int | None:
    if not status_text:
        return None
    match = re.search(r"(\d+)\s*OT", status_text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if re.search(r"\bOT\b", status_text, flags=re.IGNORECASE):
        return 1
    return None


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


def _fetch_game_result(game_id: str) -> GameResult:
    try:
        response = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id, timeout=30)
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

        home_score = home_line.get("score") if home_line else None
        away_score = away_line.get("score") if away_line else None

        overtime = _parse_game_status_overtimes(summary.get("gameStatusText"))
        if overtime is None:
            period = summary.get("period")
            try:
                period_value = int(period) if period is not None else None
            except (TypeError, ValueError):
                period_value = None
            if period_value and period_value > 4:
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
                "jerseyNum": "jerseyNum",
            },
        )

        return GameResult(
            home_score=int(home_score) if home_score is not None else None,
            away_score=int(away_score) if away_score is not None else None,
            overtimes=overtime or 0,
            attendance=attendance,
            inactive_players=inactive_players,
        )
    except Exception as exc:  # Fallback for legacy or temporary API issues.
        logger.warning("BoxScoreSummaryV3 failed for %s: %s", game_id, exc)
        response = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id, timeout=30)
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

        home_score = home_line.get("PTS") if home_line else None
        away_score = away_line.get("PTS") if away_line else None

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

        return GameResult(
            home_score=int(home_score) if home_score is not None else None,
            away_score=int(away_score) if away_score is not None else None,
            overtimes=overtime or 0,
            attendance=attendance,
            inactive_players=inactive_players,
        )


def enrich_upcoming_game_result(input_path: Path, output_dir: Path) -> Path:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_game_id = payload.get("gameId")
    if raw_game_id is None:
        raise ValueError(f"Missing gameId in {input_path}")
    # add leading zeros to make it 10 digits
    game_id = f"{raw_game_id:010d}"

    logger.info("Fetching final score for game %s", game_id)
    result = _fetch_game_result(game_id)

    payload["homeTeamFinalScore"] = result.home_score
    payload["awayTeamFinalScore"] = result.away_score
    payload["overtimes"] = result.overtimes
    payload["attendance"] = result.attendance
    payload["inactivePlayers"] = result.inactive_players

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    logger.info("Saved enriched game to %s", output_path)
    return output_path


def enrich_upcoming_games_results(
    input_dir: Path,
    output_dir: Path,
) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for input_path in sorted(input_dir.glob("*.json")):
        try:
            written.append(enrich_upcoming_game_result(input_path, output_dir))
        except Exception as exc:
            logger.warning("Skipping %s due to error: %s", input_path, exc)

    logger.info("Enriched %s upcoming games into %s", len(written), output_dir)
    return written


def main() -> None:
    setup_logging(level="INFO")
    parser = argparse.ArgumentParser(
        description="Enrich upcoming game JSON files with final scores."
    )
    parser.add_argument("input_path", type=Path, nargs="?")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_INCREMENTAL_DIR / "upcoming_games",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_INCREMENTAL_DIR / "upcoming_games_results",
    )
    args = parser.parse_args()

    if args.input_path:
        enrich_upcoming_game_result(args.input_path, args.output_dir)
    else:
        enrich_upcoming_games_results(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
