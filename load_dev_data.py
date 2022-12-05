import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import mat73


# make fake FNC data
class DevData(Dataset):
    def __init__(self,
                 N_components=53,
                 N_timepoints=490,
                 window_size=40,
                 N_subs=200,
                 sz_mu=0,
                 sz_std=3,
                 sz_low=40,
                 sz_high=80,
                 N_hc=1024,
                 hc_mu=0.1,
                 hc_std=1,
                 hc_low=20,
                 hc_high=60):
        self.N_subjects = N_subs

        upper_triangle_size = int(N_components*(N_components - 1)/2)
        ts_length = N_timepoints - window_size
        self.dim = upper_triangle_size
        self.seqlen = ts_length

        UKB_demo = pd.read_csv('/data/users2/mduda/scripts/brainAge/UKBiobank_age_gender_ses01_final.csv')
        UKB_demo_clean = UKB_demo[UKB_demo.age > 15]


        sub_data =np.zeros((N_subs, ts_length, upper_triangle_size)) 
        dfnc_path = "/data/qneuromark/Results/DFNC/UKBioBank/"
        dfnc_file = "/UKB_dfnc_sub_001_sess_001_results.mat"
        for i in range(N_subs):
            fname = dfnc_path + UKB_demo_clean.iloc[i].DFNC_filename + dfnc_file
            data = mat73.loadmat('/data/qneuromark/Results/DFNC/UKBioBank/Sub00001/UKB_dfnc_sub_001_sess_001_results.mat')
            dfnc = data['FNCdyn']
            sub_data[i,:,:] = dfnc[:ts_length, :]


        self.data = sub_data.astype('float32')
        self.age = np.array(UKB_demo_clean.iloc[:N_subs].age).reshape(N_subs, 1).astype('float32')

    def __len__(self):
        return self.N_subjects

    def __getitem__(self, k):
        return torch.from_numpy(self.data[k, ...]), torch.from_numpy(self.age[k])
