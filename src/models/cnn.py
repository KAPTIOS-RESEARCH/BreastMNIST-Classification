import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)

class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 2, base_channels: int = 32, img_size: int = 128, num_blocks: int = 3, dropout: float = 0.5):
        super().__init__()

        conv_blocks = []
        input_channels = in_channels
        
        for i in range(num_blocks):
            output_channels = base_channels * (2 ** i)
            conv_blocks.append(ConvBlock(input_channels, output_channels))
            input_channels = output_channels

        self.net = nn.Sequential(*conv_blocks)

        self.flattened_dim = self._get_flattened_shape((in_channels, img_size, img_size))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, self.flattened_dim // 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(self.flattened_dim // 3, self.flattened_dim // 6, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(self.flattened_dim // 6, out_channels, bias=False),
        )

    def _get_flattened_shape(self, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.net(dummy)
            return dummy.numel()

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x
