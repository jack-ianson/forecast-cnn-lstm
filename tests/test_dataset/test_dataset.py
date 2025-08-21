import pytest
import torch

from forecast import datasets


def test_dataset():
    """
    Example test.
    """
    dummy_data = torch.rand((100, 10, 28, 28))
    dummy_dates = torch.arange(0, 99)

    dataset1 = datasets.WeatherDataset(
        data=dummy_data, datetimes=dummy_dates, t=24, forecast_t=1
    )

    x, y, date_x, date_y = dataset1[0]

    assert x.shape[0] == 24
    assert date_y == date_x[-1] + 1
