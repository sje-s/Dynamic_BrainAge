import torch.nn as nn
import torch 
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

class ConvRNN(nn.Module):
    def __init__(self, 
                 seqlen,
                 dim,
                 rnn='lstm',  
                 hidden_size=256, 
                 out_size=1, 
                 bidirectional=True, 
                 num_layers=1, 
                 drp=0.1, 
                 hidden_activation=nn.ReLU,
                 output_activation=None,
                 hidden_neurons=[128],
                 kernels=[],
                 strides=[],
                 paddings=[],
                 dilations=[],
                 agg='mean', 
                 channel = True,
                 bias=True):
        super(ConvRNN, self).__init__()
        self.channel = channel
        mul = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.agg = agg
        mul = 2 if bidirectional else 1
        kernels = resolve_lists(kernels, hidden_neurons, DEFAULT_KERNEL)
        strides = resolve_lists(strides, hidden_neurons, DEFAULT_STRIDE)
        paddings = resolve_lists(paddings, hidden_neurons, DEFAULT_PADDING)
        dilations = resolve_lists(dilations, hidden_neurons, DEFAULT_DILATION)
        Hin = seqlen
        input_dim = dim
        if not self.channel:
            Hin = dim
            input_dim = seqlen
        in0 = input_dim
        hin = Hin
        self.layers = nn.ModuleList()
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
            hin = next_1dCNN_shape(in0, hin,
                                   kernels[i], strides[i],
                                   paddings[i], dilations[i])
            in0 = h0
        if not self.channel:
            hin, in0 = in0, hin
        if rnn == "lstm":
            self.model = nn.LSTM(in0, hidden_size, bidirectional=bidirectional, num_layers=num_layers, batch_first=True)
        elif rnn == "gru":
            self.model = nn.GRU(in0, hidden_size, bidirectional=bidirectional, num_layers=num_layers, batch_first=True)
        else:
            self.model = nn.RNN(in0, hidden_size, bidirectional=bidirectional, num_layers=num_layers, batch_first=True)        
        if self.channel:
            self.linear = nn.Linear(hin*hidden_size*mul, out_size, bias=bias)
        else:
            self.linear = nn.Linear(hin*hidden_size*mul, out_size, bias=bias)
        

    def forward(self, input):
        B, C, T = input.shape
        if self.channel:
            input = input.permute([0,2,1])
        output = input
        for layer in self.layers:
            output = layer(output)
        if self.channel:
            output = output.permute([0, 2, 1])
        output, hidden_enc = self.model(output)        
        output = output.flatten(1)
        output = self.linear(output)
        return output

        
    
if __name__=="__main__":
    import tqdm
    x = torch.randn((17,100,1343))
    y = torch.randint(0, 20, (17, 100,)).float()
    model = ConvRNN((17,100,1343), 
                    hidden_size=69,
                    hidden_neurons=[200],
                    kernels=[2],
                    strides=[1],
                    paddings=[0],
                    dilations=[1], 
                    out_size=1, 
                    rnn="lstm", 
                    bidirectional=False,
                    channel=False)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    pbar = tqdm.tqdm(range(500))
    for e in pbar:
        optim.zero_grad()
        yh = model(x)
        loss = nn.MSELoss()(y, yh)
        loss.backward()
        optim.step()
        pbar.set_description("Loss=%.3f" % loss.item())
    yh = model(x)
    print(yh.shape)
