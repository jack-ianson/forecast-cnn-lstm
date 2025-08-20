import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
import warnings
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        results_path: Path,
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

        # store results path
        self.results_path = results_path

        # state of model for checkpointing
        self._internal_state = {"current_epoch": 0, "total_epochs": 0}

    def train(self, epochs: int = 100, checkpoint_epoch: int = None):
        """
        Train the model. Handles both training and testing based
        on the datasets passed to the __init__ method.

        Args:
            epochs (int, optional): The number of epochs to train for. Defaults to 100.
            checkpoint_epoch (int, optional): Frequency of checkpointing
        """

        self._internal_state["total_epochs"] = (
            self._internal_state["total_epochs"] + epochs
        )

        # create some storages for errors per epoch
        if not hasattr(self, "training_loss"):
            self.training_loss = torch.zeros(epochs, device=self.device)
            self.testing_loss = torch.zeros(epochs, device=self.device)
        else:
            self.training_loss = F.pad(self.training_loss, (0, epochs))
            self.testing_loss = F.pad(self.testing_loss, (0, epochs))

        for epoch in tqdm(
            range(epochs),
            desc="Training...",
            total=self._internal_state["total_epochs"],
            initial=self._internal_state["current_epoch"],
            unit="epochs",
            leave=False,
        ):
            self.model.train()

            # store the loss values for the epoch
            epoch_train_loss = torch.zeros(
                len(self.train_dataloader), device=self.device
            )

            for batch_idx, batch in enumerate(
                tqdm(
                    self.train_dataloader,
                    desc=f"Epoch {self._internal_state['current_epoch']+1}...",
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

            self.training_loss[self._internal_state["current_epoch"]] = (
                epoch_train_loss.mean()
            )
            self.testing_loss[self._internal_state["current_epoch"]] = self.test(
                _epoch=self._internal_state["current_epoch"]
            )

            self._internal_state["current_epoch"] += 1

            if (
                self._internal_state["current_epoch"] != 0
                and (self._internal_state["current_epoch"]) % 10 == 0
            ):
                self.test_forecast(
                    path=self.results_path,
                    _epoch=self._internal_state["current_epoch"],
                )
                self.error_plot(path=self.results_path)

        self.training_loss = self.training_loss.cpu()
        self.testing_loss = self.testing_loss.cpu()

        print(f"{self._internal_state['total_epochs']} epochs complete!")

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

    def save_model(self, path: Path | str) -> None:
        """
        Save the trained model.

        Note: If intermediate model training state is required,
        call `checkpoint` instead.

        Args:
            path (Path | str): Path to save the model to.
        """
        path = Path(path)

        if path.is_dir():
            path = path / "model.pt"

        torch.save(self.model.state_dict(), path)

    def save_checkpoint(self, path: Path | str) -> None:
        """
        Checkpoint a model, including model state, optimiser state,
        current epoch and all previous training and testing loss
        values.

        Args:
            path (Path | str): Path do save the model to.'
            epoch (int): store the current epoch of the checkpoint
        """
        path = Path(path)

        if path.is_dir():
            path = path / f"checkpoint_{self._internal_state['current_epoch']}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimiser.state_dict(),
            "current_epoch": self._internal_state.get("current_epoch", 0),
            "total_epochs": self._internal_state.get("total_epochs", 0),
            "training_loss": getattr(self, "training_loss", None),
            "testing_loss": getattr(self, "testing_loss", None),
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path | str) -> None:
        """
        Load a model checkpoint to resume training

        Args:
            path (Path | str): Path of the checkpoint
        """
        path = Path(path)

        checkpoint: dict = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optim_state_dict"])
        self._internal_state["current_epoch"] = checkpoint.get("current_epoch", 0)
        self._internal_state["total_epochs"] = checkpoint.get("total_epochs", 0)
        self.training_loss = checkpoint.get("training_loss", None)
        self.testing_loss = checkpoint.get("testing_loss", None)

        if self.training_loss is not None:
            self.training_loss.to(self.device)
            self.testing_loss.to(self.device)

    def error_plot(self, path: Path | str) -> None:
        """
        Create an error plot for training and testing loss

        Args:
            path (Path | str): Path to save the image to
        """
        fig, ax = plt.subplots()

        training_loss = self.training_loss.cpu().numpy()
        testing_loss = self.testing_loss.cpu().numpy()

        ax.plot(
            np.arange(1, self._internal_state["current_epoch"] + 1),
            training_loss[: self._internal_state["current_epoch"]],
            label="Train",
        )
        ax.plot(
            np.arange(1, self._internal_state["current_epoch"] + 1),
            testing_loss[: self._internal_state["current_epoch"]],
            label="Test",
        )
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Mean Squared Error")
        ax.set_yscale("log")
        ax.set_title("Training Loss")

        fig.savefig(path / "error_plot.png")
        plt.close(fig)

    def test_forecast(
        self,
        path: Path | str,
        index: int = 0,
        min_temp: float = None,
        max_temp: float = None,
        _epoch: int = None,
    ) -> None:

        x, y, _, _ = self.testing_dataset[index]

        with torch.no_grad():

            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x.unsqueeze(0))

            target_img = y[0, 2:-2, 2:-2].cpu().numpy()
            predicted_img = output[0, 0, 2:-2, 2:-2].cpu().numpy()

            if min_temp is not None:
                target_img = target_img * (max_temp - min_temp) + min_temp
                predicted_img = predicted_img * (max_temp - min_temp) + min_temp

            difference = predicted_img - target_img

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))

            im_1 = ax[0].imshow(target_img, cmap="viridis")
            ax[0].set_title("True Temperature")
            cbar_1 = fig.colorbar(im_1, ax=ax[0])
            cbar_1.set_label("°C")
            ax[0].axis("off")

            im_2 = ax[1].imshow(predicted_img, cmap="viridis")
            ax[1].set_title("Predicted Temperature")
            cbar_2 = fig.colorbar(im_2, ax=ax[1])
            cbar_2.set_label("°C")
            ax[1].axis("off")

            im_3 = ax[2].imshow(difference, cmap="bwr")
            ax[2].set_title("Difference")
            cbar_3 = fig.colorbar(im_3, ax=ax[2])
            cbar_3.set_label("°C")
            ax[2].axis("off")

        fig.tight_layout()

        if _epoch is not None:
            fig.savefig(path / f"test_forecast_{_epoch}.png")
        else:
            fig.savefig(path / f"test_forecast_testing_sample_{index}.png")

        plt.close(fig)

    @property
    def trained_model(self) -> nn.Module:
        return self.model
