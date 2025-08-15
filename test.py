import matplotlib.pyplot as plt

from backend import fetch_weather_data, get_grid_data

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


df = fetch_weather_data(
    url=OPEN_METEO_URL,
    coords=(51.5072, -0.15),
    start_date="2023-01-01",
    end_date="2023-01-03",
    categories=["temperature_2m", "relative_humidity_2m", "windspeed_10m"],
)


data = get_grid_data(
    url=OPEN_METEO_URL,
    latitude_range=(50.0, 59.0),
    longitude_range=(-8.0, 2.0),
    start_date="2023-01-01",
    end_date="2023-01-02",
    category="temperature_2m",
    spacing=1,
)


fig, ax = plt.subplots()

ax.imshow(data[0, :, :])

plt.show()
