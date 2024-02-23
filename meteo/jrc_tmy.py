from datetime import datetime
import httpx
import pandas as pd
import numpy as np

from misc.util import cached_func


def main():
    pass

@cached_func
def get_jrc_tmy(latitude: float, longitude: float, start_year: int = 2005, end_year: int = 2015) -> pd.DataFrame:
    """
    Fetches historical weather data for a given site and prepares the dataframe.

    Parameters:
        latitude (float): The latitude of the site.
        longitude (float): The longitude of the site.
        start_year (int, optional): The start year for data retrieval. Defaults to 2005.
        end_year (int, optional): The end year for data retrieval. Defaults to 2015.

    Returns:
        pd.DataFrame: A dataframe containing the historical weather data.

    Raises:
        ValueError: If the latitude or longitude is out of bounds.
        SystemExit: If there's a request or server error.
    """
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError(
            "Latitude must be between -90 and 90, and longitude between -180 and 180."
        )

    # Send a GET request to the JRC API
    try:
        response = httpx.get(
            f"https://re.jrc.ec.europa.eu/api/tmy?lat={latitude}&lon={longitude}&startyear={start_year}&endyear={end_year}&outputformat=json",
            timeout=25,
        )
        response.raise_for_status()  # Raises an exception for 4XX/5XX responses
        data_json = response.json()  # Parse the JSON response here
    except httpx.RequestError as e:
        raise SystemExit(f"An error occurred while requesting {e.request.url!r}.") from e
    except httpx.HTTPStatusError as e:
        raise SystemExit(f"Error response {e.response.status_code} while requesting {e.request.url!r}.") from e

    # Parse the JSON response & reset the index with a new date range
    data = pd.DataFrame(data_json["outputs"]["tmy_hourly"])
    date_range = pd.date_range(start="2023-01-01 00:00:00", periods=8760, freq="H")

    # Directly use date_range to assign time components
    data["Hour_of_Day"] = date_range.hour
    data["Day_of_Year"] = date_range.dayofyear
    data["Week_of_Year"] = ((date_range.dayofyear - 1) // 7 + 1).astype(int)
    data["Week_of_Year"] = np.where(data["Week_of_Year"] > 52, 52, data["Week_of_Year"])  # Cap at 52 if necessary
    data["Month_of_Year"] = date_range.month

    return data


def get_hour(timestamp: str) -> int:
    """
    Extracts the hour from a timestamp.

    Parameters:
        timestamp (str): The timestamp in "%Y%m%d:%H%M" format.

    Returns:
        int: The hour extracted from the timestamp.
    """
    return datetime.strptime(timestamp, "%Y%m%d:%H%M").hour


def get_day(timestamp: str) -> int:
    """
    Extracts the day of the year from a timestamp.

    Parameters:
        timestamp (str): The timestamp in "%Y%m%d:%H%M" format.

    Returns:
        int: The day of the year extracted from the timestamp.
    """
    day_of_year = datetime.strptime(timestamp, "%Y%m%d:%H%M").timetuple().tm_yday
    return day_of_year


def get_week(timestamp: str) -> int:
    """
    Calculates the week of the year from a timestamp.

    Parameters:
        timestamp (str): The timestamp in "%Y%m%d:%H%M" format.

    Returns:
        int: The week of the year, with a maximum value of 52. This calculation
             assumes the week starts on January 1st and does not conform to ISO week date.
    """
    date_object = datetime.strptime(timestamp, "%Y%m%d:%H%M")
    day_of_year = date_object.timetuple().tm_yday

    week_of_year = np.minimum(52, (day_of_year - 1) // 7 + 1)
    return week_of_year


def get_month(timestamp: str) -> int:
    """
    Extracts the month from a timestamp.

    Parameters:
        timestamp (str): The timestamp in "%Y%m%d:%H%M" format.

    Returns:
        int: The month extracted from the timestamp.
    """
    month = datetime.strptime(timestamp, "%Y%m%d:%H%M").month
    return month


if __name__ == "__main__":
    main()
