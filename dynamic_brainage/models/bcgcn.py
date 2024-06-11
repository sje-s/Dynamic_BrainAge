# https://raw.githubusercontent.com/SCUT-Xinlab/BC-GCN/main/model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

lamda = 1/2
SE = 4
node = 112  # number of rois

class GPC(nn.Module):

    def __init__(self, in_dim, out_dim, node_dim=53):
        super(GPC, self).__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.conv = nn.Conv2d(in_dim, out_dim, (1, node_dim))
        nn.init.normal_(self.conv.weight, std=math.sqrt(2/(node_dim*in_dim+node_dim*out_dim)))

    def forward(self, x):
        batchsize = x.shape[0]

        x_c = self.conv(x)
        x_C = x_c.expand(batchsize, self.out_dim, self.node_dim, self.node_dim)
        x_R = x_C.permute(0,1,3,2)
        x = x_C+x_R

        return x

class GPC_Res(nn.Module):

    def __init__(self, in_dim, out_dim, node_dim=53):
        super(GPC_Res, self).__init__()
        self.out_dim = out_dim
        self.node_dim = node_dim
        self.conv = nn.Conv2d(in_dim, out_dim, (1, self.node_dim))
        nn.init.normal_(self.conv.weight, std=math.sqrt(2*(1-lamda)/(self.node_dim*in_dim+self.node_dim*out_dim)))
        self.convres = nn.Conv2d(in_dim, out_dim, 1)
        nn.init.normal_(self.convres.weight, std=math.sqrt(4*lamda/(in_dim+out_dim)))

    def forward(self, x):
        batchsize = x.shape[0]

        res = self.convres(x)
        x_c = self.conv(x)
        x_C = x_c.expand(batchsize, self.out_dim, self.node_dim, self.node_dim)
        x_R = x_C.permute(0,1,3,2)
        x = x_C+x_R+res

        return x

class GPC_SE(nn.Module):

    def __init__(self, in_dim, out_dim, node_dim=53):
        super(GPC_SE, self).__init__()
        self.node_dim=node_dim
        self.out_dim = out_dim
        self.conv = nn.Conv2d(in_dim, out_dim, (1, self.node_dim))
        nn.init.normal_(self.conv.weight, std=math.sqrt(2*(1-lamda)/(self.node_dim*in_dim+self.node_dim*out_dim)))
        self.convres = nn.Conv2d(in_dim, out_dim, 1)
        nn.init.normal_(self.convres.weight, std=math.sqrt(4*lamda/(in_dim+out_dim)))

        self.sed = nn.Linear(out_dim, int(out_dim/SE), False)
        self.seu = nn.Linear(int(out_dim/SE), out_dim, False)

    def forward(self, x):
        batchsize = x.shape[0]

        res = self.convres(x)
        x_c = self.conv(x)
        x_C = x_c.expand(batchsize, self.out_dim, self.node_dim, self.node_dim)
        x_R = x_C.permute(0,1,3,2)
        x = x_C+x_R

        se = torch.mean(x,(2,3))
        se = self.sed(se)
        se = F.relu(se)
        se = self.seu(se)
        se = torch.sigmoid(se)
        se = se.unsqueeze(2).unsqueeze(3)

        x = x.mul(se)
        x = x+res

        return x

class EP(nn.Module):

    def __init__(self, in_dim, out_dim, node_dim=53):
        super(EP, self).__init__()
        self.node_dim = node_dim
        self.conv = nn.Conv2d(in_dim, out_dim, (1, self.node_dim))
        nn.init.normal_(self.conv.weight, std=math.sqrt(4/(self.node_dim*in_dim+out_dim)))


    def forward(self, x):

        x = self.conv(x)

        return x

class NP(nn.Module):

    def __init__(self, in_dim, out_dim, node_dim=53):
        super(NP, self).__init__()
        self.node_dim = node_dim
        self.conv = nn.Conv2d(in_dim, out_dim, (self.node_dim, 1))
        nn.init.normal_(self.conv.weight, std=math.sqrt(4/(self.node_dim*in_dim+out_dim)))

    def forward(self, x):

        x = self.conv(x)

        return x

class BC_GCN(nn.Module):
    def __init__(self, GPC_dim_1, GPC_dim_2, GPC_dim_3, EP_dim, NP_dim, node_dim=53):
        self.node_dim = node_dim
        super(BC_GCN, self).__init__()

        print('Current model : BC_GCN')

        self.GPC_1 = GPC(1, GPC_dim_1, node_dim=node_dim)
        self.GPC_2 = GPC(GPC_dim_1, GPC_dim_2, node_dim=node_dim)
        self.GPC_3 = GPC(GPC_dim_2, GPC_dim_3, node_dim=node_dim)

        self.EP = EP(GPC_dim_3, EP_dim, node_dim=node_dim)

        self.NP = NP(EP_dim, NP_dim, node_dim=node_dim)

        self.fc = nn.Linear(NP_dim, 1)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        if len(x.shape) < 4:
            xz = torch.zeros((x.shape[0],1,self.node_dim,self.node_dim), device=x.device)
            xz[:,0,*np.triu_indices(self.node_dim,k=1)] = x
            xz = xz.permute([0,1,3,2])
            xz[:,0,*np.triu_indices(self.node_dim,k=1)] = x
            x = xz

        x = self.GPC_1(x)
        x = F.relu(x)

        x = self.GPC_2(x)
        x = F.relu(x)

        x = self.GPC_3(x)
        x = F.relu(x)

        x = self.EP(x)
        x = F.relu(x)

        x = self.NP(x)
        x = F.relu(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x

class BC_GCN_Res(nn.Module):
    def __init__(self, GPC_dim_1, GPC_dim_2, GPC_dim_3, EP_dim, NP_dim, node_dim=53):
        super(BC_GCN_Res, self).__init__()
        self.node_dim = node_dim

        print('Current model : BC_GCN_Res')

        self.GPC_1 = GPC_Res(1, GPC_dim_1, node_dim=node_dim)
        self.GPC_2 = GPC_Res(GPC_dim_1, GPC_dim_2, node_dim=node_dim)
        self.GPC_3 = GPC_Res(GPC_dim_2, GPC_dim_3, node_dim=node_dim)

        self.EP = EP(GPC_dim_3, EP_dim, node_dim=node_dim)

        self.NP = NP(EP_dim, NP_dim, node_dim=node_dim)

        self.fc = nn.Linear(NP_dim, 1)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        if len(x.shape) < 4:
            xz = torch.zeros((x.shape[0],1,self.node_dim, self.node_dim), device=x.device)
            xz[:,0,*np.triu_indices(self.node_dim,k=1)] = x
            xz = xz.permute([0,1,3,2])
            xz[:,0,*np.triu_indices(self.node_dim,k=1)] = x
            x = xz

        x = self.GPC_1(x)
        x = F.relu(x)

        x = self.GPC_2(x)
        x = F.relu(x)

        x = self.GPC_3(x)
        x = F.relu(x)

        x = self.EP(x)
        x = F.relu(x)

        x = self.NP(x)
        x = F.relu(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x

class BC_GCN_SE(nn.Module):
    def __init__(self, GPC_dim_1, GPC_dim_2, GPC_dim_3, EP_dim, NP_dim, node_dim=53):
        super(BC_GCN_SE, self).__init__()
        self.node_dim = node_dim

        print('Current model : BC_GCN_SE')

        self.GPC_1 = GPC_SE(1, GPC_dim_1, node_dim=node_dim)
        self.GPC_2 = GPC_SE(GPC_dim_1, GPC_dim_2, node_dim=node_dim)
        self.GPC_3 = GPC_SE(GPC_dim_2, GPC_dim_3, node_dim=node_dim)

        self.EP = EP(GPC_dim_3, EP_dim, node_dim=node_dim)

        self.NP = NP(EP_dim, NP_dim, node_dim=node_dim)

        self.fc = nn.Linear(NP_dim, 1)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        if len(x.shape) < 4:
            xz = torch.zeros((x.shape[0],1,self.input_dim, self.node_dim), device=x.device)
            xz[:,0,*np.triu_indices(self.node_dim,k=1)] = x
            xz = xz.permute([0,1,3,2])
            xz[:,0,*np.triu_indices(self.node_dim,k=1)] = x
            x = xz
        x = self.GPC_1(x)
        x = F.relu(x)

        x = self.GPC_2(x)
        x = F.relu(x)

        x = self.GPC_3(x)
        x = F.relu(x)

        x = self.EP(x)
        x = F.relu(x)

        x = self.NP(x)
        x = F.relu(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        #nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        #nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    train_data = torch.rand(90, 10, 53, 53)  # subjects * slices * node * node
    train_data = train_data.view(-1, 53, 53)
    train_data = train_data[torch.randperm(train_data.size(0))]  # shuffle train data
    train_data = train_data.view(15, 60, 1, 53, 53)  # batch number * batch size * 1 * node * node
    train_label = (torch.rand(90) * 800).int()  # labels range from 0 to 800 days
    train_label = train_label.unsqueeze(1).expand(90, 10).reshape(15, 60)
    train_label = train_label / 1

    test_data = torch.rand(10, 10, 53, 53)  # subjects * slices * node * node
    test_data = test_data.unsqueeze(2)
    test_label = (torch.rand(10) * 800).int()  # labels range from 0 to 800 days
    test_label = test_label.unsqueeze(1).expand(10, 10)
    test_label = test_label / 1

    criterion = nn.L1Loss()

    # net = model.BC_GCN(16, 16, 16, 64, 256)
    # net = model.BC_GCN_Res(16, 16, 16, 64, 256)
    net = BC_GCN_SE(16, 16, 16, 64, 256)
    net.apply(weights_init)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.3fM" % (total/1e6))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0)

    loss_best = 800
    loss_s_best = 800
    for epoch in range(50):

        net.train()
        running_loss = 0.0
        for i in range(train_label.size(0)):
            inputs = train_data[i]
            labels = train_label[i]
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()/train_label.size(0)*1

