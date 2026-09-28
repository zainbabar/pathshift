from functools import partial

import torch.nn as nn

# prefilled args to avoid repeated code
DefaultConv2d = partial(nn.Conv2d, kernel_size=3, stride=1, padding='same')


class CNNClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            DefaultConv2d(in_channels=3, out_channels=32), nn.ReLU(),
            DefaultConv2d(in_channels=32, out_channels=32), nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            DefaultConv2d(in_channels=32, out_channels=64), nn.ReLU(),
            DefaultConv2d(in_channels=64, out_channels=64), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),  # flatten for the fully connected layer
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),  # 7x7 pixels * 64 channels = 3136 inputs
            nn.Linear(128, n_classes),  # outputs our 9 logits
        )

    def forward(self, inputs):
        return self.layers(inputs)
