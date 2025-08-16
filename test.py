# import matplotlib.pyplot as plt

# from backend import get_grid_data


# OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# data = get_grid_data(
#     url=OPEN_METEO_URL,
#     latitude_range=(50.0, 59.0),
#     longitude_range=(-8.0, 2.0),
#     start_date="2023-01-01",
#     end_date="2023-02-01",
#     categories=["temperature_2m", "precipitation"],
#     spacing=0.4,
# )


# fig, ax = plt.subplots()

# ax.imshow(data[0, :, :, 0])

# plt.show()
