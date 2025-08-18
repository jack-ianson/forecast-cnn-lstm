import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from forecast import datasets, modules


BATCH_SIZE = 128
EPOCHS = 100

training_data, training_datetimes = datasets.load_monthly_data(
    paths=[
        r"data\training\jan_2023\data_0_5_spacing.h5",
        r"data\training\feb_2023\data_0_5_spacing.h5",
        r"data\training\mar_2023\data_0_5_spacing.h5",
        r"data\training\april_2023\data_0_5_spacing.h5",
    ]
)

testing_data, testing_datetimes = datasets.load_monthly_data(
    paths=[
        r"data\validation\sept_2023\data_0_5_spacing.h5",
    ]
)


training_data = training_data.permute(
    0, 3, 1, 2
)  # Change to (time, channels, height, width)
testing_data = testing_data.permute(0, 3, 1, 2)

# compute the min and max of each feature in the dataset
min_values = training_data.amin(dim=(0, 2, 3))
max_values = training_data.amax(dim=(0, 2, 3))

# Normalize the data
training_data = (training_data - min_values[None, :, None, None]) / (
    max_values - min_values
)[None, :, None, None]
testing_data = (testing_data - min_values[None, :, None, None]) / (
    max_values - min_values
)[None, :, None, None]


print(
    f"Training data shape: {training_data.shape}, Testing data shape: {testing_data.shape}"
)
# training_dataset = datasets.WeatherDataset(
#     training_data, datetimes=training_datetimes, t=24, forecast_t=1
# )
# testing_dataset = datasets.WeatherDataset(
#     testing_data, datetimes=testing_datetimes, t=24, forecast_t=1
# )
