import pandas as pd

EAST_CITIES = [
    "St. Louis",
    "Orlando",
    "New York",
    "Miami",
    "Detroit",
    "Buffalo",
    "Washington",
    "Milwaukee",
    "Toronto",
    "Chicago",
    "Charlotte",
    "Philadelphia",
    "Cincinnati",
    "Baltimore",
    "Cleveland",
    "New Jersey",
    "Boston",
    "Capital",
    "Atlanta",
    "Indiana",
    "Brooklyn",
]

WEST_CITIES = [
    "Los Angeles",
    "Phoenix",
    "San Antonio",
    "Golden State",
    "Seattle",
    "Portland",
    "Denver",
    "Utah",
    "Oklahoma City",
    "Sacramento",
    "San Francisco",
    "Vancouver",
    "Dallas",
    "Kansas City",
    "Kansas City-Omaha",
    "Houston",
    "Memphis",
    "New Orleans",
    "Minnesota",
    "San Diego",
]


def get_cities_conference():
    east_cities_df = pd.MultiIndex.from_product(
        [EAST_CITIES, ["East"]], names=["teamCity", "Conference"]
    ).to_frame(index=False)

    west_cities_df = pd.MultiIndex.from_product(
        [WEST_CITIES, ["West"]], names=["teamCity", "Conference"]
    ).to_frame(index=False)

    cities_conferences = pd.concat([east_cities_df, west_cities_df], axis=0)
    return cities_conferences
