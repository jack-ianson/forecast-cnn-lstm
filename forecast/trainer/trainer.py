import torch.nn as nn
import torch
from torch import Tensor
import warnings
import typing
from tqdm import tqdm

from ..datasets import WeatherDataset

_BatchType = tuple[Tensor, Tensor, list[str], list[str]]


class ForecastModelTrainer:
    """
    A trainer class to train the forecast CNN for weather prediction

    Args:
        model (nn.Module): The torch module
        training_dataset (WeatherDataset): The training dataset
        testing_dataset (WeatherDataset): The testing dataset
        device (torch.device | str, optional): the device. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 32.
        optimiser (torch.optim.Optimizer | str, optional): optimiser. Defaults to "adam".
        criterion (nn.Module | str, optional): loss function. Defaults to "mse".
        lr (float, optional): learning rate for the optimiser. Defaults to 0.001.
        padding (int, optional): padding to remove from the target input. This will
            not be used for back propagation. Defaults to 2.

    """

    def __init__(
        self,
        model: nn.Module,
        training_dataset: WeatherDataset,
        testing_dataset: WeatherDataset,
        device: torch.device | str = None,
        batch_size: int = 32,
        optimiser: torch.optim.Optimizer | str = "adam",
        criterion: nn.Module | str = "mse",
        lr: float = 0.001,
        padding: int = 2,
    ):

        # set the device and pass  the model to the device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)

        # store the datasets
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset

        # create dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            testing_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

        # define the optimiser
        if isinstance(optimiser, str):
            if optimiser.lower() == "adam":
                self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
            elif optimiser.lower() == "adamw":
                self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=lr)
            else:
                warnings.warn(f"Unknown optimiser '{optimiser}', using Adam instead.")
                self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        elif isinstance(optimiser, torch.optim.Optimizer):
            self.optimiser = optimiser
        else:
            raise ValueError(
                "Optimiser must be a string ('adam' or 'adamw') or an instance of torch.optim.Optimizer."
            )

        # define the loss function
        if isinstance(criterion, str):
            if criterion.lower() == "mse":
                self.criterion = nn.MSELoss()
            elif criterion.lower() == "l1":
                self.criterion = nn.L1Loss()
            else:
                warnings.warn(
                    f"Unknown loss function '{criterion}', using MSE instead."
                )
                self.criterion = nn.MSELoss()

        elif isinstance(criterion, nn.Module):
            self.criterion = criterion

        else:
            raise ValueError(
                "Criterion must be a string ('mse' or 'l1') or an instance of nn.Module."
            )

        # padding refers to the image pixels not to be predicted
        self.padding = padding

    def train(self, epochs: int = 100):
        """
        Train the model. Handles both training and testing based
        on the datasets passed to the __init__ method.

        Args:
            epochs (int, optional): The number of epochs to train for. Defaults to 100.
        """
        # create some storages for errors per epoch
        train_loss = torch.zeros(epochs, device=self.device)
        test_loss = torch.zeros(epochs, device=self.device)

        for epoch in tqdm(
            range(epochs), desc="Training...", total=epochs, unit="epochs"
        ):
            self.model.train()

            # store the loss values for the epoch
            epoch_train_loss = torch.zeros(
                len(self.train_dataloader), device=self.device
            )

            for batch_idx, batch in enumerate(
                tqdm(
                    self.train_dataloader,
                    desc=f"Epoch {epoch+1}...",
                    unit="steps",
                    total=len(self.train_dataloader),
                    leave=False,
                )
            ):
                # process the training batch
                loss = self._run_step(batch)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                epoch_train_loss[batch_idx] = loss.item()

            train_loss[epoch] = epoch_train_loss.mean()
            test_loss[epoch] = self.test(_epoch=epoch)

        self.training_loss = train_loss.cpu()
        self.testing_loss = test_loss.cpu()

    def test(self, _epoch: int = None) -> torch.Tensor:
        """
        Pass the testing dataset through the model. Automatically
        wraps all code with torch.no_grad()

        Args:
            _epoch (int, optional): the current epoch. Defaults to None.

        Returns:
            torch.Tensor: The average loss for the testing dataset
        """
        with torch.no_grad():

            epoch_test_loss = torch.zeros(len(self.test_dataloader), device=self.device)

            for batch_idx, batch in enumerate(
                tqdm(
                    self.test_dataloader,
                    desc=f"Epoch {_epoch+1}...",
                    unit="steps",
                    total=len(self.test_dataloader),
                    leave=False,
                )
            ):
                # process the training batch
                loss = self._run_step(batch)

                epoch_test_loss[batch_idx] = loss.item()

        return epoch_test_loss.mean()

    def _run_step(self, batch: _BatchType) -> torch.Tensor:
        """
        Helper method to process a single batch of data. Avoids
        duplication, can be used for train and test, providing it
        is called within a torch.no_grad for testing.

        Args:
            batch (_BatchType): The batch of data to process

        Returns:
            torch.Tensor: loss value
        """

        inputs, targets, _, _ = batch

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)

        # remove the outer pixels
        targets = targets[
            :, :, self.padding : -self.padding, self.padding : -self.padding
        ]
        outputs = outputs[
            :, :, self.padding : -self.padding, self.padding : -self.padding
        ]

        loss = self.criterion(outputs, targets)

        return loss
