import requests
import os
import pandas as pd
import json

from dotenv import load_dotenv
from openai import OpenAI

from pydantic import BaseModel, Field

import itertools
from tqdm import tqdm


from src.config.paths import (
    TEAMS_CITIES_LOCATIONS_HISTORY_PROCESSED_PATH,
    LOCATIONS_DISTANCES_PATH,
)
from src.config.constants import SERPER_ENDPOINT

from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")

load_dotenv(override=True)
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI()


def make_google_distance_query(city1: str, city2: str):
    return f"What is the distance in miles from {city1} to {city2}?"


def google_call(query) -> float:
    payload = {"q": query}
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    response = requests.request("POST", SERPER_ENDPOINT, headers=headers, json=payload)
    return response.text


def parse_google_search(google_response) -> str:
    resp_json = json.loads(google_response)
    relevant_context = [i["snippet"] for i in resp_json["organic"][0:3]]
    result = "\n\n".join(relevant_context)
    return result


class distance(BaseModel):
    """Distance between two cities."""

    straight_line_distance: float = Field(
        description="The straight line distance between the two cities in miles"
    )
    driving_distance: float = Field(
        description="The driving distance between the two cities in miles"
    )


def get_distances_ai(google_query, google_results_txt):
    system_prompt = """
    Your task is to extract the distance between two cities. 
    The context may have both driving distance and straight line distance (sometimes referred as flight distance, shortest distance or airline distance). 
    Get both distances if they are available, if either is missing, return the None.
    Use the following context to answer the user's question:\n\n
    """

    users_prompt = f"""
    Use the following context to answer the user's question:
    {google_results_txt}

    Question:
    {google_query}
    """

    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": "user", "content": users_prompt}
    ]

    model = "gpt-4.1-nano"
    response = openai.chat.completions.parse(
        model=model, messages=messages, response_format=distance
    )

    distance_dict = json.loads(response.choices[0].message.content)
    return distance_dict


def main():
    teams_locations = pd.read_csv(TEAMS_CITIES_LOCATIONS_HISTORY_PROCESSED_PATH)
    teams_locations["city_state"] = (
        teams_locations["city"] + ", " + teams_locations["state"]
    )

    unique_locations = list(set(teams_locations["city_state"]))
    locations_distances = []
    all_combinations = list(itertools.combinations(unique_locations, 2))
    for pair in tqdm(all_combinations[0:5]):
        logger.info(f"Processing pair: {pair[0]} to {pair[1]}")
        cities_dict = {"from": pair[0], "to": pair[1]}

        google_query = make_google_distance_query(pair[0], pair[1])
        google_response = google_call(google_query)
        google_results_txt = parse_google_search(google_response)

        distance_dict = get_distances_ai(google_query, google_results_txt)
        locations_distances.append({**cities_dict, **distance_dict})

    locations_distances_df = pd.DataFrame(locations_distances)
    locations_distances_df.to_csv(LOCATIONS_DISTANCES_PATH, index=False)


if __name__ == "__main__":
    main()
