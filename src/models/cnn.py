import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_channels),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 2, img_size: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_channels, 6, 5, 1),
            ConvBlock(6, 16, 5, 1),
        )

        self.flattened_dim = self._get_flattened_shape((img_size, img_size))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, self.flattened_dim//3, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.flattened_dim//3,
                      self.flattened_dim//5, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.flattened_dim//5, out_channels, bias=False),
        )

    def _get_flattened_shape(self, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(*input_shape)
            dummy = dummy.unsqueeze(0).unsqueeze(1)
            dummy = self.net(dummy)
            shape = dummy.shape[-3] * dummy.shape[-2] * dummy.shape[-1]
        return shape

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x
