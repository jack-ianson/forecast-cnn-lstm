import torch  #
import matplotlib.pyplot as plt

from forecast import ForecastCNN, load_h5_data, WeatherDataset


PATH = r"data\data_0_5_spacing.h5"


data, meta_data = load_h5_data(PATH, "data")


data = torch.tensor(data).float()

data = data.permute(0, 2, 1, 3)  # Change shape to (time_steps, channels, height, width)

print(f"Data shape after permute: {data.shape}")


dataset = WeatherDataset(data, t=8, forecast_t=4)

print(f"Dataset length: {len(dataset)}")


x, y = dataset[0]


print(f"Input shape: {x.shape}, Target shape: {y.shape}")


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
