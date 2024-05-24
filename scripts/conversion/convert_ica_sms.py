import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
import scipy.io
import time

TEMPLATE = {
    'SCN': [44, 52, 68, 97, 98],
 'ADN': [20, 55],
 'SMN': [1, 2, 8, 10, 26, 53, 65, 71, 79],
 'VSN': [4, 7, 11, 14, 15, 19, 61, 76, 92],
 'CON': [32, 36, 37, 42, 47, 54, 60, 62, 66, 67, 69, 78, 80, 82, 83, 87, 95],
 'DMN': [16, 22, 31, 39, 50, 70, 93],
 'CBN': [3, 6, 12, 17]
}

TEMPLATE_IDX = []
for key, val in TEMPLATE.items():
    TEMPLATE_IDX.extend(val)

class ConvertData(Dataset):
    def __init__(self,
                 N_subs=200,
                 #age_csv="/data/users3/bbaker/projects/LSTM_BrainAge/data/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores.csv",
                 age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/scripts/conversion/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
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
        self.filepath = np.array(all_data.loc[self.idxs, "SM_full_fname"])
        if 'session' in all_data.columns:
            self.session = np.array(all_data.loc[self.idxs, "session"])
        else:
            self.session = [f.split("/")[6] for f in self.filepath]

    def __len__(self):
        """Returns the leet
            i.e. the Number of subjects
        """
        return self.N_subjects

    def __getitem__(self, k):
        """Get an individual FNC matrix (flattened) and Age (integer), resolving filepath format string with index
            and using mat73 to load data
        """
        #print("Loading ", k)
        filepath = str(self.filepath[k])
        try:
            data = nib.load(filepath).get_fdata()
            data = data[:, :, :, TEMPLATE_IDX]
            return torch.from_numpy(data), filepath, torch.from_numpy(self.age[k]), self.session[k], self.subID[k]
        except Exception as e:
            return torch.zeros((1,)), filepath, torch.from_numpy(self.age[k]), self.session[k], self.subID[k]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import pandas as pd
    import tqdm
    rows = []
    test_dataset = ConvertData(N_subs=22569)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    pbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataset))
    for batch_i, (fnc, fname, age, session, subject) in pbar:
        #print(fnc.shape, fname[0])
        cfname = "/data/users3/bbaker/projects/Dynamic_BrainAge/data/sm_v2/" + fname[0][1:].replace("/","-") + ".pt"
        if not fnc.sum() == 0.:
            #if not os.path.exists(cfname):
            torch.save(fnc, cfname)
            rows.append(dict(old_filename=fname[0], new_filename=cfname, age=age, session=session[0], subject=subject[0]))        
        #break
        if batch_i % 100 == 0:
            df = pd.DataFrame(rows)
            df.to_csv('./data/sm_converted_files_v2.csv')
        pbar.update(1)
