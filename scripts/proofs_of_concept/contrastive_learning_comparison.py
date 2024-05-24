import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50
from dynamic_brainage.contrib.soft_dtw.soft_dtw_cuda import SoftDTW

class Model(nn.Module):
    def __init__(self, tree_level=4, rnn="lstm", dim=438, hidden_size=1024, bidirectional=False, num_layers=1):
        super(Model, self).__init__()
        if rnn == "lstm":
            self.model = nn.LSTM(dim, hidden_size, bidirectional=bidirectional, num_layers=num_layers)
        elif rnn == "gru":
            self.model = nn.GRU(dim, hidden_size, bidirectional=bidirectional, num_layers=num_layers)
        else:
            self.model = nn.RNN(dim, hidden_size, bidirectional=bidirectional, num_layers=num_layers)
        mul = 2 if bidirectional else 1
        # projection head
        self.g = nn.Sequential(nn.Linear(hidden_size*mul, 512, bias=False), nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True), nn.Linear(512, 1, bias=True))


    def forward(self, x):
        B, T, C = x.squeeze().shape
        x, _ = self.model(x.squeeze())
        #feature = torch.flatten(x, start_dim=1)        
        out = torch.stack([self.g(x[:,t,:]) for t in range(T)], 1)
        tree_output = torch.stack([self.tree_model(x[:,t,:]) for t in range(T)], 1)        
            
        #out = torch.stack([self.g(x[:,t,:].squeeze()) for t in range(x.shape)])
        #tree_output = self.tree_model(feature)
        return F.normalize(x, dim=-1), F.normalize(out, dim=-1), tree_output

if __name__=="__main__":
    from dynamic_brainage.dataloaders.get_dataset import get_dataset
    from tqdm import tqdm
    tree_level = 4
    net = Model(tree_level=4, dim=53, hidden_size=8, num_layers=3).cuda()
    tc_data = get_dataset("tctest",N_subs=50)
    batch_size = 2
    data_loader = torch.utils.data.DataLoader(tc_data, batch_size=batch_size)
    mean_of_probs_per_level_per_epoch = {level: torch.zeros(2**level).cuda() for level in range(1, tree_level + 1)}
    net.train()
    temperature = 1
    
    train_optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_tree_loss, total_reg_loss, total_simclr_loss = 0.0, 0.0, 0.0
    epochs = 10