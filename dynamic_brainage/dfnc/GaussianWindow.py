import numpy as np

import torch.nn as nn
import torch
import torchaudio.functional as taf

class SlidingGaussianWindow1d(nn.Module):
    def __init__(self, minTP, wsize, win_alpha=0.5, learnable=True, triu=False, device="cpu"):
        super().__init__()
        self.minTP = minTP
        self.win_alpha = torch.nn.Parameter(torch.Tensor([win_alpha]), requires_grad=learnable)
        self.register_parameter(name='win_alpha', param=self.win_alpha)
        self.wsize = wsize
        self.Nwin = minTP - wsize
        self.window_steps = list(range(self.Nwin))
        self.triu = triu
        self.nT = self.minTP
        self.nT1 = self.nT
        if self.nT % 2 != 0:
            self.nT = self.nT + 1
        self.m = self.nT/2
        self.w = int(np.round(self.wsize/2))
        self.b = torch.zeros((self.nT, 1))
        self.b[int(self.m - self.w ):int(self.m+self.w)] = 1
        self.x1 = torch.arange(self.nT).to(device)
        self.gw_init = torch.exp(- ((self.x1-self.m)**2))

    def _apply(self,fn):
        super()._apply(fn)
        self.win_alpha = fn(self.win_alpha)
        self.gw_init = fn(self.gw_init)
        self.b = fn(self.b)
        return self

    def forward(self, x):
        gw = self.gw_init / (2 * self.win_alpha * self.win_alpha).T
        A = taf.convolve(gw.squeeze(), self.b.squeeze())
        A = A/A.max()
        A = A[int(self.m):int(len(A)-self.m+1)]
        A = A[:self.nT1]
        fncs = torch.zeros((x.shape[0], self.Nwin, x.shape[2], x.shape[2])).to(x.device)
        Ashift = A.clone()
        for ii in range(self.Nwin):
            Ashift = torch.roll(Ashift, -int(np.round(-self.minTP/2) + np.round(self.wsize/2) + self.window_steps[ii] + 1))
            tcwin = x.squeeze() * Ashift.view(len(Ashift), 1)
            fncs[:, ii, :, :] = torch.stack([torch.corrcoef(tcwin[i,...].T) for i in range(tcwin.shape[0])], 0)
        return fncs
    
class FixedGaussianWindow1d(nn.Module):
    def __init__(self, minTP, wsize, win_alpha=0.5, learnable=False, triu=False, device="cpu"):
        super().__init__()
        self.minTP = minTP
        self.win_alpha = win_alpha
        self.wsize = wsize
        self.Nwin = minTP - wsize
        self.window_steps = list(range(self.Nwin))
        self.triu = triu
        self.nT = self.minTP
        self.nT1 = self.nT
        if self.nT % 2 != 0:
            self.nT = self.nT + 1
        self.m = self.nT/2
        self.w = int(np.round(self.wsize/2))
        self.b = torch.zeros((self.nT, 1))
        self.b[int(self.m - self.w ):int(self.m+self.w)] = 1
        self.x1 = torch.arange(self.nT).to(device)
        self.gw_init = torch.exp(- ((self.x1-self.m)**2))
        gw = self.gw_init / (2 * self.win_alpha * self.win_alpha)
        A = taf.convolve(gw.squeeze(), self.b.squeeze())
        A = A/A.max()
        A = A[int(self.m):int(len(A)-self.m+1)]
        A = A[:self.nT1]
        self.A = A
        Ashifts = []
        Ashift  = A.clone()
        for ii in range(self.Nwin):
            Ashift = torch.roll(Ashift, -int(np.round(-self.minTP/2) + np.round(self.wsize/2) + self.window_steps[ii] + 1))
            Ashifts.append(Ashift.clone())
        self.Ashifts = torch.stack(Ashifts,0)

    def _apply(self,fn):
        super()._apply(fn)
        self.gw_init = fn(self.gw_init)
        self.b = fn(self.b)
        self.Ashifts = fn(self.Ashifts)
        self.A = fn(self.A)
        return self

    def forward(self, x):        
        fncs = torch.zeros((x.shape[0], self.Nwin, x.shape[2], x.shape[2])).to(x.device)
        for ii in range(self.Nwin):
            tcwin = x * self.Ashifts[ii, ...].view(x.shape[1],1)
            fncs[:, ii, :, :] = torch.stack([torch.corrcoef(tcwin[i,...].T) for i in range(tcwin.shape[0])], 0)
        return fncs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sb
    import time
    N = 32
    T = 100
    C = 5
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    classes = 2
    test = torch.zeros(N*classes,T,C).to(device)
    blocks = [
        (
            ([1,1,1,1,1],[-1,-1,-1,1,1],range(int(T/2))),
            ([1,1,1,1,1],[1,1,1,-1,-1],range(int(T/4)))
        ),
        (
            ([1,1,1,1,1],[1,1,1,-1,-1],range(int(T/2),int(T/2)+int(T/4))),
            ([1,1,1,1,1],[-1,-1,-1,1,1],range(int(T/4),int(T/2)))
        ),
        (
            ([0.1,0.2,0.1,0.2,0.1],[-1,1,-1,1,-1],range(int(T/2)+int(T/4),T)),
            ([0.1,0.2,0.1,0.2,0.1],[1,-1,1,-1,1],range(int(T/2),T)),
         )
         ]
    for cblock in blocks:
        for ic, block in enumerate(cblock): 
            for c in range(C):
                for n in range(N):
                    test[(ic*N):(ic*N+N),block[-1],c] = torch.randn((len(block[-1]))).to(device)*block[0][c] + block[1][c]
    labels = torch.cat([torch.Tensor([c for i in range(N)]) for c in range(classes)], 0)
    class SimpleRNNW(nn.Module):
        def __init__(self, winsize=30, chans=5, alpha=0.5, T=100):
            super().__init__()
            self.windower = FixedGaussianWindow1d(T, winsize, alpha)
            self.rnn = nn.LSTM(15, 32, 3)
            self.mlp = nn.Linear(32*(T-winsize), 2)

        def forward(self, x):
            batch, time, chan = x.shape
            windows = self.windower(x)
            windows = windows[:,:,torch.triu(torch.ones(5, 5)) == 1]
            lstm_out, _ = self.rnn(windows)
            flat_out = torch.nn.Flatten(1)(lstm_out)
            output = self.mlp(flat_out)
            return output
    import tqdm
    model = SimpleRNNW(winsize=30, chans=C, alpha=0.5, T=T).to(device)
    initial_windows = model.windower(test).detach().cpu().numpy()
    optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if 'win_alpha' not in name], lr=1e-3)
    alpha_optimizer = torch.optim.Adam(model.windower.parameters(), lr=1e3)
    pbar = tqdm.tqdm(range(100))
    #for name, _ in model.named_parameters():
    #  print(name)
    for i in pbar:
        optimizer.zero_grad()
        alpha_optimizer.zero_grad()
        out = model(test)
        #print(out.shape)
        #print(labels.shape)
        loss = nn.CrossEntropyLoss()(out, labels.long().to(device))
        loss.backward()
        optimizer.step()
        alpha_optimizer.step()
        pbar.set_description("Loss=%f,α=%f" % (loss.item()/70, model.windower.win_alpha.item()))
        