import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from forecast import dataset, ForecastCNN


BATCH_SIZE = 128
EPOCHS = 100

training_data, training_datetimes = dataset.load_monthly_data(
    paths=[
        r"data\training\jan_2023\data_0_5_spacing.h5",
        r"data\training\feb_2023\data_0_5_spacing.h5",
        r"data\training\mar_2023\data_0_5_spacing.h5",
        r"data\training\april_2023\data_0_5_spacing.h5",
    ]
)

testing_data, testing_datetimes = dataset.load_monthly_data(
    paths=[
        r"data\validation\sept_2023\data_0_5_spacing.h5",
    ]
)


training_data = training_data.permute(
    0, 3, 1, 2
)  # Change to (time, channels, height, width)
testing_data = testing_data.permute(0, 3, 1, 2)

training_dataset = dataset.WeatherDataset(
    training_data, datetimes=training_datetimes, t=24, forecast_t=1
)
testing_dataset = dataset.WeatherDataset(
    testing_data, datetimes=testing_datetimes, t=24, forecast_t=1
)

print(
    f"Dataset sizes: {len(training_dataset)} training, {len(testing_dataset)} testing"
)

train_dataloader = DataLoader(
    training_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
test_dataloader = DataLoader(
    testing_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

cnn = ForecastCNN(
    input_channels=training_data.shape[-3], input_image_shape=(28, 28)
).to(device)

mse = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(cnn.parameters(), lr=0.0005)


train_loss = torch.zeros(EPOCHS, device=device)
test_loss = torch.zeros(EPOCHS, device=device)

for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):

    epoch_train_loss = torch.zeros(len(train_dataloader), device=device)

    for idx, batch in enumerate(
        tqdm(
            train_dataloader,
            desc="Training Batches",
            leave=False,
            total=len(train_dataloader),
            unit="batch",
        )
    ):
        inputs, targets, input_times, target_times = batch

        inputs: torch.Tensor
        targets: torch.Tensor

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = cnn(inputs)

        loss: torch.Tensor = mse(outputs, targets.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        epoch_train_loss[idx] = loss.item()

    train_loss[epoch] = epoch_train_loss.mean()

    epoch_test_loss = torch.zeros(len(test_dataloader), device=device)

    with torch.no_grad():
        for idx, batch in enumerate(
            tqdm(
                test_dataloader,
                desc="Testing Batches",
                leave=False,
                total=len(test_dataloader),
                unit="batch",
            )
        ):
            inputs, targets, input_times, target_times = batch

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = cnn(inputs)

            loss: torch.Tensor = mse(
                outputs[:, :, 2:-2, 2:-2], targets.unsqueeze(1)[:, :, 2:-2, 2:-2]
            )

            epoch_test_loss[idx] = loss.item()

    test_loss[epoch] = epoch_test_loss.mean()

    if epoch % 10 == 0:

        # some example testing data images
        with torch.no_grad():
            sample_inputs, sample_targets, _, _ = next(iter(train_dataloader))
            sample_inputs = sample_inputs.to(device)
            sample_targets = sample_targets.to(device)
            sample_outputs = cnn(sample_inputs)

            # Select first sample in batch for visualization
            idx = 0
            target_img = sample_targets[idx, 2:-2, 2:-2].cpu().squeeze().numpy()
            pred_img = sample_outputs[idx, 0, 2:-2, 2:-2].cpu().squeeze().numpy()
            diff_img = pred_img - target_img

            fig, ax = plt.subplots(2, 3, figsize=(12, 8))
            ax[0, 0].imshow(target_img, cmap="viridis")
            ax[0, 0].set_title("Training Target")
            ax[0, 1].imshow(pred_img, cmap="viridis")
            ax[0, 1].set_title("Training Prediction")
            ax[0, 2].imshow(diff_img, cmap="bwr")
            ax[0, 2].set_title("Training Difference")
            ax[0, 0].axis("off")
            ax[0, 1].axis("off")
            ax[0, 2].axis("off")

            # repeat for testing data
            sample_inputs, sample_targets, _, _ = next(iter(test_dataloader))
            sample_inputs = sample_inputs.to(device)
            sample_targets = sample_targets.to(device)
            sample_outputs = cnn(sample_inputs)
            target_img = sample_targets[idx, 2:-2, 2:-2].cpu().squeeze().numpy()
            pred_img = sample_outputs[idx, 0, 2:-2, 2:-2].cpu().squeeze().numpy()
            diff_img = pred_img - target_img
            ax[1, 0].imshow(target_img, cmap="viridis")
            ax[1, 0].set_title("Testing Target")
            ax[1, 1].imshow(pred_img, cmap="viridis")
            ax[1, 1].set_title("Testing Prediction")
            ax[1, 2].imshow(diff_img, cmap="bwr")
            ax[1, 2].set_title("Testing Difference")
            ax[1, 0].axis("off")
            ax[1, 1].axis("off")
            ax[1, 2].axis("off")
            fig.tight_layout()
            fig.savefig(f"results/t_plus_1_testing_2/epoch_{epoch+1}.png")
            plt.close()

    plt.plot(train_loss[: epoch + 1].cpu(), label="Training Loss")
    plt.plot(test_loss[: epoch + 1].cpu(), label="Testing Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss")
    plt.savefig("results/t_plus_1_testing_2/training_loss.png")
    plt.close()
