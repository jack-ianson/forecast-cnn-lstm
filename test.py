import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from forecast import dataset, ForecastCNN


data, datetimes = dataset.load_monthly_data(
    paths=[
        r"data\jan_2023\data_0_5_spacing.h5",
        r"data\feb_2023\data_0_5_spacing.h5",
        r"data\mar_2023\data_0_5_spacing.h5",
        r"data\april_2023\data_0_5_spacing.h5",
    ]
)


data = data.permute(0, 3, 1, 2)  # Change to (time, channels, height, width)
print(f"Data shape after permute: {data.shape}")


training_dataset = dataset.WeatherDataset(data, datetimes=datetimes, t=24, forecast_t=1)


print(f"Dataset length: {len(training_dataset)}")


x, y, x_time, y_time = training_dataset[0]


print(f"Input shape: {x.shape}, Target shape: {y.shape}")

print(f"Input datetimes: {x_time}, Target datetime: {y_time}")


dataloader = DataLoader(training_dataset, batch_size=512, shuffle=True, drop_last=True)


batch = next(iter(dataloader))
inputs, targets, input_times, target_times = batch
print(f"Batch inputs shape: {inputs.shape}")
print(f"Batch targets shape: {targets.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"

cnn = ForecastCNN(input_channels=data.shape[-3], input_image_shape=(28, 28)).to(device)

mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

epochs = 500

losses = []

for epoch in tqdm(range(epochs), desc="Training Epochs"):
    epoch_loss = []
    for batch in tqdm(
        dataloader,
        desc="Training Batches",
        leave=False,
        total=len(dataloader),
        unit="batch",
    ):
        inputs, targets, input_times, target_times = batch

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = cnn(inputs)

        loss = mse(outputs, targets.unsqueeze(1))  # Adjust target shape if necessary
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        epoch_loss.append(loss.detach().cpu().item())

    losses.append(sum(epoch_loss) / len(epoch_loss))

    if epoch % 10 == 0:
        with torch.no_grad():
            sample_inputs, sample_targets, _, _ = next(iter(dataloader))
            sample_inputs = sample_inputs.to(device)
            sample_targets = sample_targets.to(device)
            sample_outputs = cnn(sample_inputs)

            # Select first sample in batch for visualization
            idx = 0
            target_img = sample_targets[idx].cpu().squeeze().numpy()
            pred_img = sample_outputs[idx].cpu().squeeze().numpy()
            diff_img = pred_img - target_img

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(target_img, cmap="viridis")
            axs[0].set_title("Target")
            axs[1].imshow(pred_img, cmap="viridis")
            axs[1].set_title("Prediction")
            axs[2].imshow(diff_img, cmap="bwr")
            axs[2].set_title("Difference")
            for ax in axs:
                ax.axis("off")
            plt.suptitle(f"Epoch {epoch+1}")
            plt.savefig(f"results/t_plus_1/epoch_{epoch+1}.png")
            plt.close(fig)

    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss")
    plt.savefig("results/t_plus_1/training_loss.png")
    plt.close()
