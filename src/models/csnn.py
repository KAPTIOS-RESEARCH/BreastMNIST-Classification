import torch.nn as nn
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from src.models.base import BaseJellyNet
import torch

class SpikingConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, v_threshold: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            layer.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=v_threshold),

            layer.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=v_threshold),
            layer.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)

class CSNN(BaseJellyNet):
    def __init__(self, 
                in_channels: int = 1, 
                out_channels: int = 2, 
                n_steps=5, 
                encoding_method='direct', 
                img_size: int = 128, 
                num_blocks: int = 3, 
                base_channels: int = 32,
                dropout: float = 0.5, 
                spike_threshold: float = 0.5):
        super().__init__(in_channels, out_channels, n_steps, encoding_method)

        conv_blocks = []
        input_channels = in_channels
        
        for i in range(num_blocks):
            output_channels = base_channels * (2 ** i)
            conv_blocks.append(SpikingConvBlock(input_channels, output_channels, v_threshold=spike_threshold))
            input_channels = output_channels

        self.net = nn.Sequential(*conv_blocks)

        self.flattened_dim = self._get_flattened_shape((in_channels, img_size, img_size))

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(self.flattened_dim, self.flattened_dim // 3, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=spike_threshold),
            layer.Dropout(dropout),
            
            layer.Linear(self.flattened_dim // 3, self.flattened_dim // 6, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=spike_threshold),
            layer.Dropout(dropout),
            
            layer.Linear(self.flattened_dim // 6, out_channels, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=spike_threshold),
        )

        functional.set_step_mode(self, step_mode='m')

    def _get_flattened_shape(self, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.net(dummy)
            return dummy.numel()

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x = self.encode_input(x)
        x = self.net(x)
        x = self.classifier(x)
        x = x.mean(0)
        return x