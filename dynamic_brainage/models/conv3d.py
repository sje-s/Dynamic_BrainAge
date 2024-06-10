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


def make3dparam(param):
    if type(param) is not list:
        param = [param, param, param]
    return param


def index_CNN_shape(S_in, kernel, stride, padding, dilation):
    return int(np.floor(((S_in + 2 * padding
                        - dilation*(kernel - 1) - 1)
                         / stride) + 1))


def next_3dCNN_shape(C_in, D_in, H_in, W_in,
                     kernel, stride, padding, dilation):
    kernel = make3dparam(kernel)
    stride = make3dparam(stride)
    padding = make3dparam(padding)
    dilation = make3dparam(dilation)
    D_out = index_CNN_shape(D_in, kernel[0], stride[0],
                            padding[0], dilation[0])
    H_out = index_CNN_shape(H_in, kernel[1], stride[1],
                            padding[1], dilation[1])
    W_out = index_CNN_shape(W_in, kernel[2], stride[2],
                            padding[2], dilation[2])
    return D_out, H_out, W_out


class CNN3D(nn.Module):
    def __init__(self,
                 input_shape,
                 output_dim,
                 hidden_activation=nn.ReLU,
                 output_activation=None,
                 hidden_activation_linear=nn.ReLU,
                 output_activation_linear=None,
                 maxpool_after=False,
                 hidden_neurons=[],
                 kernels=[],
                 strides=[],
                 paddings=[],
                 dilations=[],
                 hidden_neurons_linear=[],
                 flatten=False,
                 bias=True):
        """Standard 3DCNN with
        """
        super().__init__()
        self.layers = nn.ModuleList()
        if flatten:
            self.layers.append(nn.Flatten(1))
        if len(hidden_neurons_linear) == 0:
            hidden_neurons_linear.append(output_dim)
        else:
            hidden_neurons_linear.append(output_dim)
        kernels = resolve_lists(kernels, hidden_neurons, DEFAULT_KERNEL)
        strides = resolve_lists(strides, hidden_neurons, DEFAULT_STRIDE)
        paddings = resolve_lists(paddings, hidden_neurons, DEFAULT_PADDING)
        dilations = resolve_lists(dilations, hidden_neurons, DEFAULT_DILATION)
        _, input_dim, Din, Hin, Win = input_shape
        in0 = input_dim
        din = Din
        hin = Hin
        win = Win
        for i, h0 in enumerate(hidden_neurons):
            layer = nn.Conv3d(in0, h0, kernels[i], strides[i], paddings[i],
                              dilations[i], bias=bias)
            self.layers.append(layer)
            if i != len(hidden_neurons) - 1:
                if hidden_activation is not None:
                    self.layers.append(hidden_activation())
            else:
                if output_activation is not None:                    
                    self.layers.append(output_activation())
                if maxpool_after:
                    layer = nn.MaxPool3d(kernels[i],stride=strides[i],padding=paddings[i],dilation=dilations[i])
                    din, hin, win = next_3dCNN_shape(in0, din, hin, win, kernels[i], strides[i], paddings[i], dilations[i])
                    self.layers.append(layer)
            din, hin, win = next_3dCNN_shape(in0, din, hin, win,
                                             kernels[i], strides[i],
                                             paddings[i], dilations[i])
            in0 = h0
        for i, h0 in enumerate(hidden_neurons_linear):
            if i == 0:
                self.layers.append(nn.Flatten(1))
                in0 = din*hin*win*in0
            layer = nn.Linear(in0, h0, bias=bias)
            self.layers.append(layer)
            if i != len(hidden_neurons) - 1:
                if hidden_activation_linear is not None:
                    self.layers.append(hidden_activation_linear())
            else:
                if output_activation_linear is not None:
                    self.layers.append(output_activation_linear(dim=1))
            in0 = h0

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    import tqdm
    X1 = torch.randn(256, 1, 64, 64, 64)*1 + 1
    X2 = torch.rand(256, 1, 64, 64, 64)*1 - 1
    Y1 = torch.zeros((256,))
    Y2 = torch.ones((256,))
    Xs = torch.cat([X1, X2], 0)
    Ys = torch.cat([Y1, Y2], 0)
    model = CNN3D((256, 1, 64, 64, 64), 2, hidden_neurons=[2, 3], maxpool_after=True,
                  hidden_neurons_linear=[128, 64, 32, 16, 8, 4], output_activation=nn.ReLU)
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
