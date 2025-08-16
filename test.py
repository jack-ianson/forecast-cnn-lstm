import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from forecast import dataset, ForecastCNN


stacked_data, datetimes = dataset.load_monthly_data(
    paths=[
        r"data\jan_2023\data_0_5_spacing.h5",
        r"data\feb_2023\data_0_5_spacing.h5",
        r"data\mar_2023\data_0_5_spacing.h5",
        r"data\april_2023\data_0_5_spacing.h5",
    ]
)


data = stacked_data.permute(
    0, 2, 1, 3
)  # Change shape to (time_steps, channels, height, width)

print(f"Data shape after permute: {data.shape}")


dataset = dataset.WeatherDataset(data, datetimes=datetimes, t=24, forecast_t=12)

print(f"Dataset length: {len(dataset)}")


x, y, x_time, y_time = dataset[0]


print(f"Input shape: {x.shape}, Target shape: {y.shape}")

print(f"Input datetimes: {x_time}, Target datetime: {y_time}")


dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


batch = next(iter(dataloader))
inputs, targets, input_times, target_times = batch
print(f"Batch inputs shape: {inputs.shape}")
print(f"Batch targets shape: {targets.shape}")

# cnn = ForecastCNN(
#     input_channels=len(meta_data["categories"]), input_image_shape=(26, 26)
# )

# print(f"Data shape: {data.shape}")

# six_hours = torch.tensor(data[:6, :, 2:, :]).float().unsqueeze(0)  # Add batch dimension

# print(f"Six hours shape: {six_hours.shape}")

# # change the shape to (batch_size, time_steps, channels, height, width)
# six_hours = six_hours.permute(0, 1, 4, 2, 3)

# print(f"Input shape: {six_hours.shape}")

# output = cnn(six_hours)

# print(f"Output shape: {output.shape}")


# fig, ax = plt.subplots(1, 7, figsize=(20, 3))

# for i in range(6):
#     ax[i].imshow(six_hours[0, i, 0, :, :].detach().numpy(), cmap="viridis")
#     ax[i].set_title(f"Hour {i + 1}")
#     ax[i].axis("off")

# ax[6].imshow(output[0, 0, :, :].detach().numpy(), cmap="viridis")

# plt.show()
