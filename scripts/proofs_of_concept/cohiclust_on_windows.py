import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50
from dynamic_brainage.contrib.soft_dtw.soft_dtw_cuda import SoftDTW

def probability_vec_with_level(feature, level):
        prob_vec = torch.tensor([], requires_grad=True).cuda()
        for u in torch.arange(2**level-1, 2**(level+1) - 1, dtype=torch.long):
            probability_u = torch.ones_like(feature[:, 0], dtype=torch.float32).cuda()
            while(u > 0):
                if u/2 > torch.floor(u/2):
                    # Go left
                    u = torch.floor(u/2) 
                    u = u.long()
                    probability_u *= feature[:, u]
                elif u/2 == torch.floor(u/2):
                    # Go right
                    u = torch.floor(u/2) - 1
                    u = u.long()
                    probability_u *=  1 - feature[:, u]
            prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(1)), dim=1)
        return prob_vec

def tree_loss(tree_output1, tree_output2, batch_size, mask_for_level, mean_of_probs_per_level_per_epoch, tree_level):
    ## TREE LOSS
    loss_value = torch.tensor([0], dtype=torch.float32, requires_grad=True).cuda()

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()
    
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels * ~mask
    out_tree = torch.cat([tree_output1, tree_output2], dim=0)

    for level in range(1, tree_level + 1):
        prob_features = probability_vec_with_level(out_tree, level)
        prob_features = prob_features * mask_for_level[level]
        if level == tree_level:
            mean_of_probs_per_level_per_epoch[tree_level] += torch.mean(prob_features, dim=0)
        # Calculate loss on positive classes
        # To avoid nan while calculating sqrt https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702  https://github.com/richzhang/PerceptualSimilarity/issues/69
        loss_value -= torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels > 0)[0]].unsqueeze(1) +  1e-8), torch.sqrt(prob_features[torch.where(labels > 0)[1]].unsqueeze(2) + 1e-8))))
        # Calculate loss on negative classes
        loss_value += torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels == 0)[0]].unsqueeze(1) + 1e-8), torch.sqrt(prob_features[torch.where(labels == 0)[1]].unsqueeze(2) + 1e-8))))
    return loss_value

def regularization_loss(tree_output1, tree_output2,  masks_for_level, tree_level):
    out_tree = torch.cat([tree_output1, tree_output2], dim=0)
    loss_reg = torch.tensor([0], dtype=torch.float32, requires_grad=True).cuda()
    for level in range(1, tree_level+1):
        prob_features = probability_vec_with_level(out_tree, level)
        probability_leaves = torch.mean(prob_features, dim=0)
        probability_leaves_masked = masks_for_level[level] * probability_leaves
        for leftnode in range(0,int((2**level)/2)):
            if not (masks_for_level[level][2*leftnode] == 0 or masks_for_level[level][2*leftnode+1] == 0):
                loss_reg -=   (1/(2**level)) * (0.5 * torch.log(probability_leaves_masked[2*leftnode]) + 0.5 * torch.log(probability_leaves_masked[2*leftnode+1]))
    return loss_reg
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
        self.g = nn.Sequential(nn.Linear(hidden_size*mul, 128, bias=False), nn.BatchNorm1d(128),
                                nn.ReLU(inplace=True), nn.Linear(128, 64, bias=True))
        self.tree_model = nn.Sequential(nn.Linear(hidden_size*mul, ((2**(tree_level+1))-1) - 2**tree_level), nn.Sigmoid())
        self.masks_for_level = {level: torch.ones(2**level).cuda() for level in range(1, tree_level+1)}


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
    net = Model(tree_level=4, dim=1378, hidden_size=8, num_layers=3).cuda()
    tc_data = get_dataset("ukbhcp_v2",N_subs=999999999999999999999, N_timepoints=200)
    batch_size = 32
    data_loader = torch.utils.data.DataLoader(tc_data, batch_size=batch_size, num_workers=8, prefetch_factor=2, drop_last=True)
    mean_of_probs_per_level_per_epoch = {level: torch.zeros(2**level).cuda() for level in range(1, tree_level + 1)}
    net.train()
    temperature = .5
    
    train_optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    total_loss, total_num = 0.0, 0
    total_tree_loss, total_reg_loss, total_simclr_loss = 0.0, 0.0, 0.0

    epochs = 100
    pretrain = 90
    for epoch in range(epochs):
        print("Epoch ", epoch)
        train_bar = tqdm(data_loader)        
        for pos_1, target in train_bar:
            train_optimizer.zero_grad()
            pos_1 = pos_1.cuda(non_blocking=True)
            feature_1, out_1, tree_output1 = net(pos_1)
            # [2*B, D]
            out = out_1.view(out_1.shape[0]*out_1.shape[1], out_1.shape[2])
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(out.shape[0], device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(out.shape[0], -1)
            # compute loss
            pos_sim = torch.exp(torch.sum(out * out, dim=-1) / temperature)
            # [2*B]
            #pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss_simclr = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            ##
            train_optimizer.zero_grad()
            tree_loss_value = 0.
            regularization_loss_value = 0.
            if epoch >= pretrain:
                N,T,C = tree_output1.shape
                tree_loss_value = tree_loss(tree_output1.view(N*T,C), tree_output1.view(N*T,C), batch_size*T, net.masks_for_level, mean_of_probs_per_level_per_epoch, tree_level)
                regularization_loss_value = regularization_loss(tree_output1.view(N*T,C), tree_output1.view(N*T,C), net.masks_for_level, tree_level)
                loss = loss_simclr + tree_loss_value + (2**(-tree_level)*regularization_loss_value)
            else:
                loss = loss_simclr
                tree_loss_value = torch.zeros([1]) # Don't calculate the loss
                regularization_loss_value = torch.zeros([1]) # Don't calculate the loss 
            loss.backward()
            train_optimizer.step()

            total_num += batch_size
            total_tree_loss += tree_loss_value.item() * batch_size
            total_reg_loss += regularization_loss_value.item() * batch_size
            total_simclr_loss += loss_simclr.item() * batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        torch.save(net.state_dict(), "/data/users3/bbaker/projects/Dynamic_BrainAge/scripts/proofs_of_concept/treecluster.pt")
        print("Saved")
