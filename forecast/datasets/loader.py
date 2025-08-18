import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import warnings
import torch


def load_monthly_data(paths: list[str]):

    data_list = []

    time_steps = []

    for path in paths:

        data, metadata = load_h5_data(path, "data")

        data_list.append(torch.tensor(data))

        start_date = metadata["start_date"]
        end_date = metadata["end_date"]
        n_time_steps = data.shape[0]

        # Generate a list of datetime objects for each time step in this file
        time_range = pd.date_range(start=start_date, periods=n_time_steps, freq="h")
        time_steps.extend(time_range.to_list())

    stacked_data = torch.cat(data_list, dim=0)

    # permute to (time, channels, height, width)
    stacked_data = stacked_data.permute(0, 3, 1, 2)

    time_steps = [str(ts) for ts in time_steps]

    return stacked_data.float(), time_steps


def load_h5_data(file_path: str, dataset_name: str) -> tuple[np.ndarray, dict]:

    with h5py.File(file_path, "r") as f:

        dataset = f[dataset_name]
        data = dataset[:]

        metadata = {key: dataset.attrs[key] for key in dataset.attrs.keys()}
    return data, metadata
