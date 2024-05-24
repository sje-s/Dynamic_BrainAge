import numpy as np

def gaussianwindow(N, x0, sigma):
    x = np.arange(N)
    w = np.exp(- ((x-x0)**2)/ (2 * sigma * sigma)).T
    return w

def compute_sliding_window(nT, win_alpha, wsize):
    nT1 = nT
    if nT % 2 != 0:
        nT = nT + 1
    m = nT/2
    w = int(np.round(wsize/2))
    gw = gaussianwindow(nT, m, win_alpha)
    b = np.zeros((nT, 1))
    b[int(m -w):int(m+w)] = 1
    print(gw.shape, b.shape)
    c = np.convolve(gw.squeeze(), b.squeeze())
    c = c/max(c)
    c = c[int(m):int(len(c)-m+1)]
    c = c[:nT1]
    return c

def window_mask(Ashift2, stepNo, window_alpha, wsize):
  nT = len(Ashift2)
  mask_windows = 1
  zero_val = 1e-4
  tmp_mask = np.zeros((nT, 1)) 
  print("math", np.round((window_alpha/2))*wsize)
  print(stepNo)
  minTp = max([0, stepNo - np.round((window_alpha/2)*wsize)])
  maxTp = min([nT, stepNo + np.round((window_alpha/2))*wsize])
  print(minTp, maxTp)
  tmp_mask[int(minTp):int(maxTp)] = 1
  #print(tmp_mask)
  msk_inds = np.argwhere(tmp_mask == 1)
  # print(msk_inds)
  idx = np.where(Ashift2[msk_inds] <= zero_val)
  #print('idx', idx)
  return msk_inds

if __name__=="__main__":
    import scipy.io as sio  
    Xf = sio.loadmat("/data/users3/bbaker/projects/pydfnc/data/fbirnp3_rest_C100_ica_TC_scrubbed_filt_RSN.mat")['TC_rsn_filt']
    Xf = np.transpose(Xf, [1, 2, 0])
    wsize= 30
    minTP = 159
    window_alpha = 3
    A = compute_sliding_window(minTP, window_alpha, wsize)
    Nwin = minTP - wsize
    window_steps = list(range(Nwin))
    windows = np.zeros((Xf.shape[0], Nwin, minTP-1, Xf.shape[2]))
    fncs = np.zeros((Xf.shape[0], Nwin, Xf.shape[2], Xf.shape[2]))
    Ashift = A
    for b in range(Xf.shape[0]):
        X = Xf[0, ...].squeeze()
        Xwin = list()
        for ii in range(Nwin):
            Ashift = np.roll(A, int(np.round(-minTP/2) + np.round(wsize/2) + window_steps[ii])+1)
            #msk = window_mask(Ashift, window_steps[ii], window_alpha, wsize)
            msk = list(range(len(Ashift)))
            tcwin = X[msk, ...] * Ashift[msk].reshape(len(Ashift[msk]), 1)
            Xwin.append(tcwin)            
            corrs = np.corrcoef(tcwin)
            fncs[b, ii, :, :] = corrs    
            