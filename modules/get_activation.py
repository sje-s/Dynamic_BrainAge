import torch.nn as nn
from modules.linear_activation import LinearActivation
MODELS = dict(
    linear=LinearActivation,
    tanh=nn.Tanh,
    sigmoid=nn.Sigmoid,
    relu=nn.ReLU,
    lrelu=nn.LeakyReLU
)

def get_activation(name):
    if name not in MODELS.keys():
        raise(NotImplementedError("%s not implemented in activations" % name))
    return MODELS[name]