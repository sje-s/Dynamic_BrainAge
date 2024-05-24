import torch.nn as nn
import torch 

class SimpleSeqRNN(nn.Module):
    def __init__(self, seqlen, dim, rnn='lstm',  hidden_size=256, out_size=1, bidirectional=True, num_layers=1, drp=0.1, agg='mean', bias=True):
        super(SimpleSeqRNN, self).__init__()
        mul = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.agg = agg
        mul = 2 if bidirectional else 1
        if rnn == "lstm":
            self.model = nn.LSTM(dim, hidden_size, bidirectional=bidirectional, num_layers=num_layers, batch_first=True)
        elif rnn == "gru":
            self.model = nn.GRU(dim, hidden_size, bidirectional=bidirectional, num_layers=num_layers, batch_first=True)
        else:
            self.model = nn.RNN(dim, hidden_size, bidirectional=bidirectional, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size*mul, out_size, bias=bias)

    def forward(self, input):
        output, hidden_enc = self.model(input)        
        full_out = torch.stack([self.linear(output[:,t,:].squeeze()) for t in range(output.shape[1])], -1)
        return full_out.squeeze()
    
if __name__=="__main__":
    import tqdm
    x = torch.randn((17,100,1343))
    y = torch.randint(0, 20, (17, 100,)).float()
    model = SimpleSeqRNN(100, 1343, hidden_size=32, out_size=1, rnn="lstm", bidirectional=False)
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
