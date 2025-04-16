import pandas as pd
from datetime import datetime
from utils import get_nba_season

EAST_CITIES = [
    "Orlando",
    "New York",
    "Miami",
    "Detroit",
    "Washington",
    "Milwaukee",
    "Chicago",
    "Philadelphia",
    "Cleveland",
    "Boston",
    "Atlanta",
    "Indiana",
    "Charlotte",
    "New Jersey",
]

WEST_CITIES = [
    "Los Angeles",
    "Phoenix",
    "San Antonio",
    "Golden State",
    "Portland",
    "Denver",
    "Utah",
    "Seatle",
    "Sacramento",
    "Dallas",
    "Houston",
    "Minnesota",
]


def create_cities_conference():
    east_cities, west_cities = EAST_CITIES.copy(), WEST_CITIES.copy()
    # === EAST CITIES ===
    # Toronto Added
    start_date = "1995-07-01"
    end_date = "2002-07-01"
    east_cities.append("Toronto")
    date_range = pd.date_range(start=start_date, end=end_date, freq="YE")
    east_cities_1995_2001 = pd.MultiIndex.from_product(
        [east_cities, date_range, ["East"]],
        names=["teamCity", "ye_date", "conference"],
    ).to_frame(index=False)

    # Charlotte --> New Orleans
    start_date = "2002-07-01"
    end_date = "2004-07-01"
    east_cities.remove("Charlotte")
    east_cities.append("New Orleans")
    date_range = pd.date_range(start=start_date, end=end_date, freq="YE")
    east_cities_2002_2003 = pd.MultiIndex.from_product(
        [east_cities, date_range, ["East"]],
        names=["teamCity", "ye_date", "conference"],
    ).to_frame(index=False)

    # Charlotte Re-Added and New Orleans changed conference
    start_date = "2004-07-01"
    end_date = "2011-07-01"
    east_cities.remove("New Orleans")
    east_cities.append("Charlotte")
    date_range = pd.date_range(start=start_date, end=end_date, freq="YE")
    east_cities_2004_2011 = pd.MultiIndex.from_product(
        [east_cities, date_range, ["East"]],
        names=["teamCity", "ye_date", "conference"],
    ).to_frame(index=False)

    # New Jersey --> Broolyn
    start_date = "2012-07-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    east_cities.remove("New Jersey")
    east_cities.append("Brooklyn")
    date_range = pd.date_range(start=start_date, end=end_date, freq="YE")
    east_cities_2012_2100 = pd.MultiIndex.from_product(
        [east_cities, date_range, ["East"]],
        names=["teamCity", "ye_date", "conference"],
    ).to_frame(index=False)

    # === WEST CITIES ===
    # Vancouver Added
    start_date = "1995-07-01"
    end_date = "2001-07-01"
    west_cities.append("Vancouver")
    date_range = pd.date_range(start=start_date, end=end_date, freq="YE")
    west_cities_1995_2000 = pd.MultiIndex.from_product(
        [west_cities, date_range, ["West"]],
        names=["teamCity", "ye_date", "conference"],
    ).to_frame(index=False)

    # Vancouver --> Memphis
    start_date = "2001-07-01"
    end_date = "2003-07-01"
    west_cities.remove("Vancouver")
    west_cities.append("Memphis")
    date_range = pd.date_range(start=start_date, end=end_date, freq="YE")
    west_cities_2001_2003 = pd.MultiIndex.from_product(
        [west_cities, date_range, ["West"]],
        names=["teamCity", "ye_date", "conference"],
    ).to_frame(index=False)

    # Charlote --> New Orleans
    start_date = "2004-07-01"
    end_date = "2005-07-01"
    west_cities.append("New Orleans")
    date_range = pd.date_range(start=start_date, end=end_date, freq="YE")
    west_cities_2004_2004 = pd.MultiIndex.from_product(
        [west_cities, date_range, ["West"]],
        names=["teamCity", "ye_date", "conference"],
    ).to_frame(index=False)

    # New Orleans with Oklahoma City -- Katrina
    start_date = "2005-07-01"
    end_date = "2007-07-01"
    west_cities.append("Oklahoma City")
    date_range = pd.date_range(start=start_date, end=end_date, freq="YE")
    west_cities_2005_2006 = pd.MultiIndex.from_product(
        [west_cities, date_range, ["West"]],
        names=["teamCity", "ye_date", "conference"],
    ).to_frame(index=False)

    # New Orleans full time
    start_date = "2007-07-01"
    end_date = "2008-07-01"
    west_cities.remove("Oklahoma City")
    date_range = pd.date_range(start=start_date, end=end_date, freq="YE")
    west_cities_2007_2007 = pd.MultiIndex.from_product(
        [west_cities, date_range, ["West"]],
        names=["teamCity", "ye_date", "conference"],
    ).to_frame(index=False)

    # Seatle --> Oklahoma City
    start_date = "2009-07-01"
    west_cities.remove("Seatle")
    west_cities.append("Oklahoma City")
    end_date = datetime.today().strftime("%Y-%m-%d")
    date_range = pd.date_range(start=start_date, end=end_date, freq="YE")
    west_cities_2008_2100 = pd.MultiIndex.from_product(
        [west_cities, date_range, ["West"]],
        names=["teamCity", "ye_date", "conference"],
    ).to_frame(index=False)

    # === Concat ===
    cities_conferences = pd.concat(
        [
            east_cities_1995_2001,
            east_cities_2002_2003,
            east_cities_2004_2011,
            east_cities_2012_2100,
            west_cities_1995_2000,
            west_cities_2001_2003,
            west_cities_2004_2004,
            west_cities_2005_2006,
            west_cities_2007_2007,
            west_cities_2008_2100,
        ],
        axis=0,
    )

    cities_conferences["season"] = cities_conferences["ye_date"].apply(get_nba_season)
    # cities_conferences = cities_conferences.drop(columns=["ye_date"])
    return cities_conferences


if __name__ == "__main__":
    cities_conferences = create_cities_conference()
    cities_conferences.sort_values(
        by=["conference", "teamCity", "ye_date"],
        ascending=[True, True, True],
        inplace=True,
    )

    cities_conferences.to_csv(
        "../data/processed/teams_cities_conferences.csv",
        index=False,
    )
    print(cities_conferences)
