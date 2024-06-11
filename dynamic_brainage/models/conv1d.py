import torch
import torch.nn as nn
import numpy as np
DEFAULT_KERNEL = 3
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 0
DEFAULT_DILATION = 1


def resolve_lists(a, b, val):
    while len(a) < len(b):
        a.append(val)
    return a


def make1dparam(param):
    if type(param) is not list:
        param = [param]
    return param


def index_CNN_shape(S_in, kernel, stride, padding, dilation):
    return int(np.floor(((S_in + 2 * padding
                        - dilation*(kernel - 1) - 1)
                         / stride) + 1))


def next_1dCNN_shape(C_in, H_in,
                     kernel, stride, padding, dilation):
    kernel = make1dparam(kernel)
    stride = make1dparam(stride)
    padding = make1dparam(padding)
    dilation = make1dparam(dilation)
    H_out = index_CNN_shape(H_in, kernel[0], stride[0],
                            padding[0], dilation[0])
    return H_out


class CNN1D(nn.Module):
    def __init__(self,
                 Hin,
                 input_dim,
                 output_dim=1,
                 hidden_activation=nn.ReLU,
                 output_activation=nn.ReLU,
                 hidden_activation_linear=nn.ReLU,
                 output_activation_linear=None,
                 hidden_neurons=[128],
                 kernels=[],
                 strides=[],
                 paddings=[],
                 dilations=[],
                 hidden_neurons_linear=[128],
                 flatten=False,
                 maxpool_after=False,
                 bias=True):
        """Standard 3DCNN with
        """
        super().__init__()
        self.layers = nn.ModuleList()
        if flatten:
            self.layers.append(nn.Flatten(1))
        hidden_neurons_linear.append(output_dim)
        kernels = resolve_lists(kernels, hidden_neurons, DEFAULT_KERNEL)
        strides = resolve_lists(strides, hidden_neurons, DEFAULT_STRIDE)
        paddings = resolve_lists(paddings, hidden_neurons, DEFAULT_PADDING)
        dilations = resolve_lists(dilations, hidden_neurons, DEFAULT_DILATION)
        in0 = input_dim
        hin = Hin
        for i, h0 in enumerate(hidden_neurons):
            layer = nn.Conv1d(in0, h0, kernels[i], strides[i], paddings[i],
                              dilations[i], bias=bias)
            self.layers.append(layer)
            if i != len(hidden_neurons) - 1:
                if hidden_activation is not None:
                    self.layers.append(hidden_activation())
            else:
                if output_activation is not None:
                    self.layers.append(output_activation())
                if maxpool_after:
                    layer = nn.MaxPool1d(kernels[i],stride=strides[i],padding=paddings[i],dilation=dilations[i])
                    hin = next_1dCNN_shape(in0, hin, kernels[i], strides[i], paddings[i], dilations[i])
                    self.layers.append(layer)
            hin = next_1dCNN_shape(in0, hin,
                                   kernels[i], strides[i],
                                   paddings[i], dilations[i])
            in0 = h0
        for i, h0 in enumerate(hidden_neurons_linear):
            if i == 0:
                self.layers.append(nn.Flatten(1))
                in0 = hin*in0
            layer = nn.Linear(in0, h0, bias=bias)
            self.layers.append(layer)
            if i != len(hidden_neurons_linear) - 1:
                if hidden_activation_linear is not None:
                    self.layers.append(hidden_activation_linear())
            else:
                if output_activation_linear is not None:
                    self.layers.append(output_activation_linear())
            in0 = h0

    def forward(self, x):
        x = x.permute([0,2,1])
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    import tqdm
    X1 = torch.randn(64, 448, 1378)*1 + 1
    X2 = torch.rand(64, 448, 1378)*1 - 1
    Y1 = torch.zeros((64,))
    Y2 = torch.ones((64,))
    Xs = torch.cat([X1, X2], 0)
    Ys = torch.cat([Y1, Y2], 0)
    model = CNN1D(448,1378, output_dim=1, hidden_neurons=[1024,512,256],
                  hidden_neurons_linear=[])
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    pbar = tqdm.tqdm(range(100))
    for epoch in pbar:
        idx = torch.randperm(64)
        X = Xs[idx, ...]
        Y = Ys[idx]
        opt.zero_grad()
        Yh = model(X)
        loss = nn.CrossEntropyLoss()(Yh, Y.long())
        loss.backward()
        opt.step()
        pbar.set_description("Loss=%.20f" % loss.item())
