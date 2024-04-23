import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
import scipy.io
import time

class ConvertData(Dataset):
    def __init__(self,
                 N_subs=200,
                 age_csv="/data/users3/bbaker/projects/LSTM_BrainAge/data/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores.csv",
                 ):
        """DevData - Load FNCs for UKBiobank
        KWARGS:
            N_components    int     the number of ICs
            N_timepoints    int     the length of the scan
            window_size     int     window size for dFNC
            N_subs          int     number of subjects
            data_root       str     root directory for loading subject data
            subj_form       strf    format string for resolving subject folders
            data_file       str     the filename for loading individual subject data
        """
        self.N_subjects = N_subs
        all_data = pd.read_csv(age_csv)
        #self.seqlen = N_cut
        #np.random.seed(319)
        self.idxs = np.arange(len(all_data))
        self.age = np.array(all_data.loc[self.idxs, "age"]).reshape(
            N_subs, 1).astype('float32')
        self.subID = np.array(all_data.loc[self.idxs, "Subject"])
        self.filepath = np.array(all_data.loc[self.idxs, "sMRI_full_fname"])

    def __len__(self):
        """Returns the length of the dataset
            i.e. the Number of subjects
        """
        return self.N_subjects

    def __getitem__(self, k):
        """Get an individual FNC matrix (flattened) and Age (integer), resolving filepath format string with index
            and using mat73 to load data
        """
        #print("Loading ", k)
        filepath = self.filepath[k]
        data = nib.load(filepath).get_fdata()
        return torch.from_numpy(data), filepath, torch.from_numpy(self.age[k])


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import pandas as pd
    import tqdm
    rows = []
    test_dataset = ConvertData(N_subs=15885)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    for batch_i, (fnc, fname, age) in tqdm.tqdm(enumerate(test_dataloader)):
        #print(fnc.shape, fname[0])
        cfname = "/data/users3/bbaker/projects/LSTM_BrainAge/data/smri/" + fname[0][1:].replace("/","-") + ".pt"
        torch.save(fnc, cfname)
        rows.append(dict(old_filename=fname[0], new_filename=cfname, age=age))
        #break
        df = pd.DataFrame(rows)
        df.to_csv('./data/smri_converted_files.csv')
