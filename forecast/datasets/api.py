from tqdm import tqdm
import numpy as np
import pandas as pd
import requests
import warnings


def fetch_weather_data(
    url: str, coords: tuple, start_date: str, end_date: str, categories: list
) -> pd.DataFrame:

    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": categories,
        "timezone": "Europe/London",
    }

    response = requests.get(url, params=params)

    if response.status_code == 429:
        warnings.warn(
            "Rate limit exceeded. Please try again later or reduce the number of requests."
        )
        return pd.DataFrame(columns=categories)

    data = response.json()

    try:
        return pd.DataFrame(data["hourly"]).set_index("time").astype(float)
    except KeyError:
        warnings.warn(
            f"Data for coordinates {coords} not found in the response. "
            "Check if the coordinates are valid or if the data is available."
        )
        return pd.DataFrame(columns=categories)


def get_grid_data(
    url: str,
    latitude_range: tuple,
    longitude_range: tuple,
    start_date: str,
    end_date: str,
    categories: str,
    spacing: float = 0.1,
) -> np.ndarray:

    latitudes = np.arange(latitude_range[0], latitude_range[1], spacing)
    longitudes = np.arange(longitude_range[0], longitude_range[1], spacing)

    # create the numpy array to hold the data of shape (hours, latitudes, longitudes)

    test = fetch_weather_data(
        url=url,
        coords=(latitudes[0], longitudes[0]),
        start_date=start_date,
        end_date=end_date,
        categories=categories,
    )

    num_hours = len(test)

    data = np.zeros((num_hours, len(longitudes), len(latitudes), len(categories)))

    pb = tqdm(
        total=len(latitudes) * len(longitudes),
        desc=f"Fetching data",
        unit="grid point",
    )

    failed_coords = []

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            df = fetch_weather_data(
                url=url,
                coords=(lat, lon),
                start_date=start_date,
                end_date=end_date,
                categories=categories,
            )
            if df.empty:
                warnings.warn(
                    f"No data found for coordinates ({lat}, {lon}) in the specified date range."
                )

                data[:, i, j] = np.zeros((num_hours, len(categories)))
                failed_coords.append((lat, lon))
            else:

                data[:, i, j] = df[categories].values

            pb.update(1)

    return data, failed_coords
