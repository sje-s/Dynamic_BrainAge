from scipy.io import loadmat, savemat
import numpy as np
import sys

# loadmat()
data = loadmat("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")["DFNC_FBIRN"]
bal = int(sys.argv[1])

pad1 = np.zeros((bal, data[0][0].shape[1]))
pad2 = np.zeros((311 - bal, data[0][0].shape[1]))

for i in range(data.shape[0]):

    data[i][0] = np.concatenate((pad1, data[i][0], pad2), axis=0)\
    
savemat("paddedBothFBIRN.mat", {"DFNC_FBIRN": data})
