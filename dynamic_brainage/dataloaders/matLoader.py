import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat


# make fake FNC data
class MatLoader(Dataset):
    def __init__(self,
                 N_components=53,
                 N_timepoints=448,
                 dataLocation="/home/users/sedwardsswart/Documents/DFNC/Dynamic_BrainAge-master/paddedFBIRN.mat",
                 dataVar="DFNC_FBIRN",
                 ageLocation="/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat",
                 ageVar = "analysis_SCORE",
                 ageCol = 0,
                 typ="N/A",
                 inds="N/A"):
        matFile = loadmat(dataLocation)
        ageFile = loadmat(ageLocation)
        N_timepoints = matFile[dataVar][0][0].shape[0]

        self.N_subjects = matFile[dataVar].shape[0]

        data = np.zeros((self.N_subjects, N_timepoints, 1378))
        for i in range(self.N_subjects):
            if(typ=="front"):
                data[i, (448-matFile[dataVar][i][0].shape[0]):, :] = matFile[dataVar][i][0]
            elif(typ=="back"):
                data[i, :matFile[dataVar][i][0].shape[0], :] = matFile[dataVar][i][0]
            elif(typ=="both"):
                temp = int((448 - matFile[dataVar][i][0].shape[0])/2)
                data[i, temp:(temp+matFile[dataVar][i][0].shape[0]), :] = matFile[dataVar][i][0]
            else:
                data[i, :, :] = matFile[dataVar][i][0]

        if (inds == "True"):
            ind = loadmat("/home/users/sedwardsswart/Documents/DFNC/UKBBandMDDindexes.mat")["mddIndex"]
            ind = ind - 1
            data = np.zeros((self.N_subjects, 165, 1378))
            for i in range(self.N_subjects):
                data[i] = matFile[dataVar][i][0][ind]

        upper_triangle_size = int(N_components*(N_components - 1)/2)
        self.dim = upper_triangle_size
        self.seqlen = N_timepoints

        self.data = np.array(data, dtype=np.float32)
        self.age = np.array(ageFile[ageVar][:, ageCol], dtype=int)

        temp = np.where(~np.isnan(self.age))
        self.age = self.age[temp]
        self.data = np.squeeze(self.data[temp, :, :])
        self.N_subjects = self.age.shape[0]

        temp = np.where(self.age > 0)
        self.age = self.age[temp]
        self.data = np.squeeze(self.data[temp, :, :])
        self.N_subjects = self.age.shape[0]

    def __len__(self):
        return self.N_subjects

    def __getitem__(self, k):
        return torch.from_numpy(self.data[k, ...]), torch.from_numpy(np.array(self.age[k])), k