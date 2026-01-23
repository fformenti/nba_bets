"""LLM-assisted agents for incremental ingestion."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Iterable

import pandas as pd

from src.utils.logging_config import get_logger
from .schema import CANONICAL_GAME_COLUMNS, DEFAULT_GAME_VALUES, REQUIRED_GAME_COLUMNS

logger = get_logger(__name__)


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized.strip()


CANONICAL_SYNONYMS = {
    "gameId": ["game_id", "id", "gameid"],
    "gameDate": ["game_date", "date", "gamedate", "game_datetime"],
    "hometeamCity": ["home_city", "homecity", "home_team_city"],
    "hometeamName": ["home_name", "homename", "home_team_name"],
    "hometeamId": ["home_team_id", "hometeamid", "homeid", "home_teamid"],
    "awayteamCity": ["away_city", "awaycity", "away_team_city"],
    "awayteamName": ["away_name", "awayname", "away_team_name"],
    "awayteamId": ["away_team_id", "awayteamid", "awayid", "away_teamid"],
    "homeScore": ["home_score", "homescore", "home_pts", "homepoints"],
    "awayScore": ["away_score", "awayscore", "away_pts", "awaypoints"],
    "winner": ["winner_id", "winnerid", "winning_team_id"],
    "gameType": ["game_type", "type", "gametype"],
    "attendance": ["att", "attendance_count"],
    "arenaId": ["arena_id", "arenaid", "venue_id"],
    "gameLabel": ["game_label", "label"],
    "gameSubLabel": ["game_sublabel", "sub_label", "sublabel"],
    "seriesGameNumber": ["series_game_number", "seriesgamenumber", "series_no"],
}


def build_heuristic_mapping(columns: Iterable[str]) -> dict[str, str]:
    normalized_map = {
        normalize_column_name(col): col for col in columns if isinstance(col, str)
    }
    mapping: dict[str, str] = {}

    canonical_lookup = {
        normalize_column_name(col): col for col in CANONICAL_GAME_COLUMNS
    }
    synonym_lookup = {
        normalize_column_name(variant): canonical
        for canonical, variants in CANONICAL_SYNONYMS.items()
        for variant in variants
    }

    for normalized, original in normalized_map.items():
        if normalized in canonical_lookup:
            mapping[original] = canonical_lookup[normalized]
            continue
        if normalized in synonym_lookup:
            mapping[original] = synonym_lookup[normalized]
            continue
        close = get_close_matches(normalized, canonical_lookup.keys(), n=1, cutoff=0.85)
        if close:
            mapping[original] = canonical_lookup[close[0]]

    return mapping


@dataclass
class LLMResult:
    raw_text: str
    mapping: dict[str, str]


class LLMClient:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIChatClient(LLMClient):
    def __init__(self, model: str, timeout_seconds: int = 30):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.endpoint = os.getenv(
            "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
        )

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            return ""

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You map dataset columns to a target schema. "
                        "Return only JSON mapping from source column to canonical column."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            return response_payload["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("LLM request failed: %s", exc)
            return ""


class ColumnMappingAgent:
    def __init__(self):
        self.logger = logger

    def map_columns(self, df: pd.DataFrame) -> dict[str, str]:
        mapping = build_heuristic_mapping(df.columns)
        self.logger.info("Heuristic column mapping: %s", mapping)
        return mapping


class LLMColumnMappingAgent(ColumnMappingAgent):
    def __init__(self, llm_client: LLMClient | None = None, use_llm: bool = False):
        super().__init__()
        self.llm_client = llm_client
        self.use_llm = use_llm

    def map_columns(self, df: pd.DataFrame) -> dict[str, str]:
        mapping = super().map_columns(df)
        if not self.use_llm or self.llm_client is None:
            return mapping

        prompt = (
            "Canonical columns:\n"
            f"{CANONICAL_GAME_COLUMNS}\n\n"
            "Source columns:\n"
            f"{list(df.columns)}\n\n"
            "Return JSON mapping from source to canonical. "
            "Only include keys that map to a canonical column."
        )
        raw = self.llm_client.complete(prompt)
        llm_mapping = self._parse_llm_mapping(raw)
        if llm_mapping:
            mapping.update(llm_mapping)
            self.logger.info("LLM column mapping added: %s", llm_mapping)
        return mapping

    def _parse_llm_mapping(self, raw_text: str) -> dict[str, str]:
        try:
            parsed = json.loads(raw_text)
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {
            key: value
            for key, value in parsed.items()
            if value in CANONICAL_GAME_COLUMNS
        }


class RowNormalizationAgent:
    def __init__(self):
        self.logger = logger

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()

        for column in CANONICAL_GAME_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = DEFAULT_GAME_VALUES.get(column)

        normalized["gameDate"] = pd.to_datetime(normalized["gameDate"], errors="coerce")
        for column in ["gameId", "hometeamId", "awayteamId", "winner", "arenaId"]:
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        for column in ["homeScore", "awayScore", "attendance", "seriesGameNumber"]:
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        normalized = self._fill_winner(normalized)
        return normalized

    def _fill_winner(self, df: pd.DataFrame) -> pd.DataFrame:
        if "winner" not in df.columns:
            return df
        missing = (
            df["winner"].isna() & df["homeScore"].notna() & df["awayScore"].notna()
        )
        df.loc[missing & (df["homeScore"] > df["awayScore"]), "winner"] = df.loc[
            missing & (df["homeScore"] > df["awayScore"]), "hometeamId"
        ]
        df.loc[missing & (df["homeScore"] < df["awayScore"]), "winner"] = df.loc[
            missing & (df["homeScore"] < df["awayScore"]), "awayteamId"
        ]
        return df


class QualityAssuranceAgent:
    def __init__(self):
        self.logger = logger

    def validate(self, df: pd.DataFrame) -> list[str]:
        issues: list[str] = []
        missing_required = [
            col for col in REQUIRED_GAME_COLUMNS if col not in df.columns
        ]
        if missing_required:
            issues.append(f"Missing required columns: {missing_required}")

        null_required = [
            col
            for col in REQUIRED_GAME_COLUMNS
            if col in df.columns and df[col].isna().any()
        ]
        if null_required:
            issues.append(f"Null values detected in columns: {null_required}")

        if issues:
            for issue in issues:
                self.logger.warning("QA issue: %s", issue)
        return issues
