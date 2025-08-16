import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastCNN(nn.Module):
    def __init__(self, input_channels: int, input_image_shape: tuple = (28, 28)):
        super(ForecastCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        dummy_input = torch.zeros(1, input_channels, *input_image_shape)

        output_shape: torch.Tensor = self.cnn(dummy_input)
        output_shape = output_shape.view(1, -1).shape[1]

        self.lstm = nn.LSTM(
            input_size=output_shape,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(128, 64 * 7 * 7)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, t, c, h, w = x.size()

        cnn_features = []

        for i in range(t):
            features = self.cnn(x[:, i])
            features = features.reshape((b, -1))
            cnn_features.append(features)

        cnn_features = torch.stack(cnn_features, dim=1)

        lstm_out, _ = self.lstm(cnn_features)

        lstm_out = lstm_out[:, -1, :]  # Take the last time step output

        output = self.fc(lstm_out)  #

        output = output.view(b, 64, 7, 7)

        output = self.decoder_cnn(output)

        return output
