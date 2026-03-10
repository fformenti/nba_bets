import os
import pandas as pd
import json

from dotenv import load_dotenv
from openai import OpenAI

from pydantic import BaseModel, Field

from enum import Enum

from tqdm import tqdm


from src.config.paths import (
    TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
    TEAMS_CITIES_LOCATIONS_HISTORY_PROCESSED_PATH,
)

from src.config.constants import list_of_nba_cities, list_of_nba_states

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

load_dotenv(override=True)
openai = OpenAI()


# Create Enums from lists - the way to get "Literal-like" validation from dynamic lists
NBACityEnum = Enum("NBACityEnum", {c: c for c in list_of_nba_cities})
NBAStateEnum = Enum("NBAStateEnum", {s: s for s in list_of_nba_states})


class NBALocation(BaseModel):
    """Retrieve city and state given an NBA team name."""

    city: NBACityEnum = Field(description="The city where the team is located")
    state: NBAStateEnum = Field(description="The state where the team is located")


class NBATeam(NBALocation):
    """Class to hold a NBA team name and its city and state."""

    team_name: str = Field(description="The name of the NBA team")


def get_city_and_state(team_name: str) -> NBALocation:
    system_prompt = """
    I will give you an NBA team name and you will respond with the city and state where the team is located or was located. \n\n
    """
    message = f"Team name: {team_name}"
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": "user", "content": message}
    ]

    model = "gpt-4.1-nano"
    response = openai.chat.completions.parse(
        model=model, messages=messages, response_format=NBALocation
    )

    return json.loads(response.choices[0].message.content)


def main():
    teams_cities_history = pd.read_csv(TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH)
    teams_cities_history["teamFullName"] = (
        teams_cities_history["teamCity"] + " " + teams_cities_history["teamName"]
    )
    teams_names_cities_df = (
        teams_cities_history[["teamId", "teamFullName"]].drop_duplicates().copy()
    )
    teams_names_cities = teams_names_cities_df.to_dict(orient="records")

    teams_names_locations = []
    for team_name in tqdm(teams_names_cities):
        city_state_info = get_city_and_state(team_name)
        teams_names_locations.append({**team_name, **city_state_info})
    teams_names_locations_df = pd.DataFrame(teams_names_locations)

    teams_cities_locations_history = teams_cities_history.merge(
        teams_names_locations_df, on=["teamId", "teamFullName"], how="left"
    )

    teams_cities_locations_history.to_csv(
        TEAMS_CITIES_LOCATIONS_HISTORY_PROCESSED_PATH, index=False
    )


if __name__ == "__main__":
    main()
    print("Done")
    print("--------------------------------")
    print("--------------------------------")
