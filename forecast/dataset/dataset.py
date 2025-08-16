import torch
from torch import Tensor
from torch.utils.data import Dataset


class WeatherDataset(Dataset):
    def __init__(self, data: torch.Tensor, t: int = 24, forecast_t: int = 8):
        """
        Initializes the WeatherDataset with the provided data.

        Args:
            data (torch.Tensor): A tensor containing weather data.
            t (int): The number of time steps in the input sequence.
            forecast_t (int): The time to the forecast.
        """
        self.data = data
        self.t = t
        self.forecast_t = forecast_t

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.data.shape[0] - self.t - self.forecast_t + 1

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        """
        Retrieves a sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the input sequence and the target sequence.
        """
        x = self.data[index : index + self.t]

        y = self.data[index + self.t + self.forecast_t]

        return x, y
