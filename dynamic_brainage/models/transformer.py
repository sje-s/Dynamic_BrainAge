import torch
import torch.nn as nn


class MyTransformer(nn.Module):

    def __init__(self, seqlen, dim,
                 hidden_size=128, 
                 bidirectional=True, 
                 drp=0.1, 
                 nhead=2,
                 num_layers=1):
        """
        Seqlen - length of the sequence
        Dim - dimension of the input sequence
        Hidden_size - hidden_size of the feed forward component
        """
        super(MyTransformer, self).__init__()        
        self.hidden_size = hidden_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(seqlen*dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.outlayer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_lstm = self.transformer_encoder(x)
        h_flat = torch.flatten(h_lstm, 1)
        out = self.relu(self.linear(h_flat))
        out = self.dropout(out)
        out = self.outlayer(out)
        return out


if __name__=="__main__":
    X = torch.randn(32, 100, 1378)
    model = MyTransformer(100, 1378, hidden_size=128, nhead=53)
    yh = model(X)
    print(yh.shape)
    def count_parameters(mmodel):
        return sum(p.numel() for p in mmodel.parameters() if p.requires_grad)
    print(count_parameters(model))
