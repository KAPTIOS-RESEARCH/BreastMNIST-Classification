import torch.nn as nn
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from .base import BaseJellyNet
import torch

class SConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2), 
        )
    
    def forward(self, x):
        return self.block(x)        


class CSNN(BaseJellyNet):
    def __init__(self, in_channels: int = 1, out_channels: int = 2, n_steps=5, encoding_method='direct', img_size : int = 128):
        super().__init__(in_channels, out_channels, n_steps, encoding_method)

        self.net = nn.Sequential(
            SConvBlock(1, 64, 3, 1),
            SConvBlock(64, 128, 3, 1),
            SConvBlock(128, 128, 3, 1),
        )

        self.flattened_dim = self._get_flattened_shape((img_size, img_size))
        
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(self.flattened_dim, 2000, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            
            layer.Linear(2000, out_channels, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')

    def _get_flattened_shape(self, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(*input_shape)
            dummy = dummy.unsqueeze(0).unsqueeze(1)
            dummy = self.net(dummy)
            shape = dummy.shape[-3] * dummy.shape[-2] * dummy.shape[-1]
        return shape

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x = self.encode_input(x)
        x = self.net(x)
        x = self.classifier(x)
        x = x.mean(0)
        return x