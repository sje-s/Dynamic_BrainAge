import torch
import torch.nn as nn


class BiLSTM(nn.Module):

    def __init__(self, seqlen, dim, hidden_size=128, bidirectional=True, drp=0.1, num_layers=3):
        """
        Seqlen - length of the sequence
        Dim - dimension of the input sequence
        Hidden_size - hidden_size of the feed forward component
        """
        super(BiLSTM, self).__init__()
        #drp = 0.1
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(dim, self.hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(seqlen)
        self.linear = nn.Linear(seqlen*hidden_size*2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.outlayer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm = self.batchnorm(h_lstm)
        h_flat = torch.flatten(h_lstm, 1)
        out = self.relu(self.linear(h_flat))
        out = self.dropout(out)
        out = self.outlayer(out)
        return out
if __name__=="__main__":
    X = torch.randn(32, 100, 1378)
    model = BiLSTM(100, 1378, hidden_size=128)
    yh = model(X)
    print(yh.shape)
    def count_parameters(mmodel):
        return sum(p.numel() for p in mmodel.parameters() if p.requires_grad)
    print(count_parameters(model))