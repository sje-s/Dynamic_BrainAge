import torch.nn as nn
import torch

class SeqRNN(nn.Module):
    def __init__(self, seqlen, dim, rnn_enc='lstm', rnn_dec='lstm', hidden_size_enc=256, hidden_size_dec=256, out_size=1, bidirectional=True, num_layers=1, attention=True, attention_width=3, drp=0.1):
        super(SeqRNN, self).__init__()
        mul = 2 if bidirectional else 1
        if rnn_enc.lower() == 'lstm':
            self.encoder = nn.LSTM(dim, hidden_size_enc, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif rnn_enc.lower() == 'gru':
            self.encoder = nn.GRU(dim, hidden_size_enc, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        else:
            self.encoder = nn.RNN(dim, hidden_size_enc, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.attention = None
        if attention:
            self.attention = nn.Linear(hidden_size_enc*mul, hidden_size_enc*mul)
        if rnn_dec.lower() == 'lstm':
            self.decoder = nn.LSTM(hidden_size_enc*mul, hidden_size_dec, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif rnn_dec.lower() == 'gru':
            self.decoder = nn.GRU(hidden_size_enc*mul, hidden_size_dec, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        else:
            self.decoder = nn.RNN(hidden_size_enc*mul, hidden_size_dec, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)        
        self.out = nn.Linear(hidden_size_dec*mul, out_size)
        self.attention_width = attention_width

    def forward(self, input):
        output, hidden_enc = self.encoder(input)
        if self.attention is not None:
            output = torch.stack([self.attention(output[:,t,:].squeeze()) for t in range(output.shape[1])], 1)
        #output = self.drp(output)
        output, hidden_dec = self.decoder(output)
        output = torch.stack([self.out(output[:,t,:].squeeze()) for t in range(output.shape[1])], 1)
        return output.squeeze()

if __name__ == "__main__":
    x = torch.randn(32, 100, 1343).to('cuda')
    model = SeqRNN(100, 1343, hidden_size_enc=32, hidden_size_dec=32, out_size=1, rnn_enc="gru", rnn_dec="lstm").to('cuda')
    y = model(x)
    print(y.shape)

        
        