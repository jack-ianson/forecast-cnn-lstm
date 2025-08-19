import torch
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pformat
import argparse


from forecast import datasets, modules
from forecast.trainer import ForecastModelTrainer


def main(
    data_path: Path,
    results_path: Path,
    epochs: int,
    batch_size: int,
    checkpoint: Path = None,
):

    results_path.mkdir(exist_ok=True, parents=True)

    training_paths = list(Path(data_path / "training").glob("**/*.h5"))
    testing_paths = list(Path(data_path / "validation").glob("**/*.h5"))

    print("Setting up training run....")
    print("-------------------------------------")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Training paths: {pformat(training_paths, indent=2)}")
    print(f"Testing paths: {pformat(testing_paths, indent=2)}")
    print("-------------------------------------\n")

    print("Loading training and testing data...")
    training_data, training_datetimes = datasets.load_monthly_data(paths=training_paths)
    testing_data, testing_datetimes = datasets.load_monthly_data(paths=testing_paths)

    print("Training data shape (time, features, height, width):", training_data.shape)
    print("Testing data shape (time, features, height, width):", testing_data.shape)

    # compute the min and max of each feature in the dataset
    min_values = training_data.amin(dim=(0, 2, 3))
    max_values = training_data.amax(dim=(0, 2, 3))

    print("Normalizing data between 0 and 1...\n")
    # Normalize the data
    training_data = (training_data - min_values[None, :, None, None]) / (
        max_values - min_values
    )[None, :, None, None]
    testing_data = (testing_data - min_values[None, :, None, None]) / (
        max_values - min_values
    )[None, :, None, None]

    # create the datasets
    training_dataset = datasets.WeatherDataset(
        training_data, datetimes=training_datetimes, t=24, forecast_t=1
    )
    testing_dataset = datasets.WeatherDataset(
        testing_data, datetimes=testing_datetimes, t=24, forecast_t=1
    )

    # create model
    cnn = modules.ForecastCNN(
        input_channels=training_data.shape[-3], input_image_shape=(28, 28)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the trainer
    trainer = ForecastModelTrainer(
        model=cnn,
        training_dataset=training_dataset,
        testing_dataset=testing_dataset,
        optimiser="adamw",
        criterion="mse",
        batch_size=batch_size,
        device=device,
        results_path=results_path,
    )

    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    trainer.train(epochs=epochs)

    trainer.save_checkpoint(path=results_path)
    trainer.save_model(path=results_path)

    trainer.error_plot(path=results_path)

    trainer.test_forecast(
        path=results_path, min_temp=min_values[0].item(), max_temp=max_values[0].item()
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", "-p", help="path to the datasets", type=str, required=True
    )
    parser.add_argument(
        "--results_path", help="path to the results", type=str, required=True
    )
    parser.add_argument("--checkpoint", type=str, help="path to a model checkpoint")
    parser.add_argument(
        "--epochs", type=int, default=100, help="The number of epochs to train for"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for training"
    )

    args = parser.parse_args()

    main(
        data_path=Path(args.data_path),
        results_path=Path(args.results_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint=Path(args.checkpoint) if args.checkpoint is not None else None,
    )
