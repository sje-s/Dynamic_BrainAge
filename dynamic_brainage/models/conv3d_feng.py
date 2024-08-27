import torch
import torch.nn as nn
from dynamic_brainage.models.conv3d import CNN3D

class Feng(CNN3D):
    def __init__(self, input_shape, output_dim):
        super().__init__(input_shape, output_dim, hidden_activation=None, output_activation=nn.ReLU, maxpool_after=True)