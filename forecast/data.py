import requests
import numpy as np
import pandas as pd
from tqdm import tqdm


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
    data = response.json()

    return pd.DataFrame(data["hourly"]).set_index("time").astype(float)


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
        coords=(1.0, 1.0),
        start_date=start_date,
        end_date=end_date,
        categories=categories,
    )

    num_hours = len(test)

    data = np.zeros((num_hours, len(latitudes), len(longitudes), len(categories)))

    pb = tqdm(
        total=len(latitudes) * len(longitudes),
        desc=f"Fetching data",
        unit="grid point",
    )

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            df = fetch_weather_data(
                url=url,
                coords=(lat, lon),
                start_date=start_date,
                end_date=end_date,
                categories=categories,
            )
            data[:, i, j] = df[categories].values

            pb.update(1)

    return data
