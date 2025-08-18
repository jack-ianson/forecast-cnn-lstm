import torch
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pformat


from forecast import datasets, modules
from forecast.trainer import ForecastModelTrainer


BATCH_SIZE = 128
EPOCHS = 10
CRITERION = "mse"
OPTIMISER = "adamw"

TRAINING_PATHS = list(Path(r"data\training").glob("**/*.h5"))
TESTING_PATHS = list(Path(r"data\validation").glob("**/*.h5"))


print("Setting up training run....")
print("-------------------------------------")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Criterion: {CRITERION}")
print(f"Optimiser: {OPTIMISER}")
print(f"Training paths: {pformat(TRAINING_PATHS, indent=2)}")
print(f"Testing paths: {pformat(TESTING_PATHS, indent=2)}")
print("-------------------------------------\n")


print("Loading training and testing data...")
training_data, training_datetimes = datasets.load_monthly_data(paths=TRAINING_PATHS)
testing_data, testing_datetimes = datasets.load_monthly_data(paths=TESTING_PATHS)


print("Training data shape (time, features, height, width):", training_data.shape)
print("Testing data shape (time, features, height, width):", testing_data.shape)

# compute the min and max of each feature in the dataset
min_values = training_data.amin(dim=(0, 2, 3))
max_values = training_data.amax(dim=(0, 2, 3))

print("Normalizing data between 0 and 1...")
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

x_test, y_test, _, _ = training_dataset[0]
print(f"Sample input shape: {x_test.shape}, sample target shape: {y_test.shape}")

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
    optimiser=OPTIMISER,
    criterion=CRITERION,
    batch_size=BATCH_SIZE,
    device=device,
)


trainer.train(epochs=EPOCHS)


plt.plot(trainer.training_loss.numpy(), label="Training Loss")
plt.plot(trainer.testing_loss.numpy(), label="Testing Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training Loss")
plt.savefig("results/t_plus_t_transformed/training_loss.png")
plt.close()
