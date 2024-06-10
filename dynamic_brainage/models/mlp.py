import torch.nn as nn
from dynamic_brainage.modules.linear_activation import LinearActivation
from dynamic_brainage.modules.get_activation import get_activation


class MLP(nn.Module):
    def __init__(self,
                 m_in=128,
                 m_out=128,
                 h=[],
                 hidden_activation=LinearActivation,
                 hidden_activation_kwargs={},
                 output_activation=LinearActivation,
                 output_activation_kwargs={},
                 flatten=False,
                 unflatten=False):
        super(MLP, self).__init__()
        next_i = m_in
        next_o = m_out
        h.append(m_out)
        self.layers = nn.ModuleList([])
        if type(hidden_activation) is str:
            hidden_activation = get_activation(hidden_activation)

        if type(output_activation) is str:
            output_activation = get_activation(output_activation)
        if flatten:
            self.layers.append(nn.Flatten(1))
        for i, next_o in enumerate(h):
            if i > 0:
                if hidden_activation != LinearActivation:
                    self.layers.append(hidden_activation(
                        **hidden_activation_kwargs))
            layer = nn.Linear(next_i, next_o, bias=False)
            next_i = next_o
            self.layers.append(layer)
        if output_activation != LinearActivation:
            self.layers.append(output_activation(**output_activation_kwargs))
        self.unflatten = unflatten

    def forward(self, x):
        S = x.shape
        for layer in self.layers:
            x = layer(x)
        if self.unflatten:
            x = x.reshape(*S)
        return x
