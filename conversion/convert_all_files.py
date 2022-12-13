import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import mat73
import scipy.io
import time

class ConvertData(Dataset):
    def __init__(self,
                 N_components=53,
                 N_timepoints=448,
                 N_cut=20,
                #  window_size=40,
                 N_subs=200,
                 #data_root="/data/qneuromark/Results/DFNC/UKBioBank",
                 #subj_form="%s", #"Sub%05d",
                 #data_file="UKB_dfnc_sub_001_sess_001_results.mat",
                 age_csv="/data/users2/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths.csv",
                #  age_threshold=15
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
        # The dFNC filepath for each subject is a format string, where the root and filename stay the same
        # but subject directory changes
        #self.filepath_form = os.path.join(data_root, subj_form, data_file)
        
        # Compute the size of the upper-triangle in the FNC matrix
        upper_triangle_size = int(N_components*(N_components - 1)/2)
        # ts_length = N_timepoints - window_size
        # These variables are useful for properly defining the RNN used
        self.dim = upper_triangle_size
        self.seqlen = N_timepoints
        # Load demographic data (small enough to keep in memory)
        # UKB_demo = pd.read_csv(age_csv)
        # UKB_demo_clean = UKB_demo[UKB_demo.age > age_threshold]
        all_data = pd.read_csv(age_csv)
        self.seqlen = N_cut
        np.random.seed(319)
        self.idxs = np.random.permutation(len(all_data))[:N_subs]
        self.age = np.array(all_data.loc[self.idxs, "age"]).reshape(
            N_subs, 1).astype('float32')
        self.subID = np.array(all_data.loc[self.idxs, "Subject"])
        self.filepath = np.array(all_data.loc[self.idxs, "DFNC_full_fname"])

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
        start = time.time()
        filepath = self.filepath[k]
        try:
            data = mat73.loadmat(filepath)
            #print("\tmat73")
        except:
            data = scipy.io.loadmat(filepath)
            #print("\tscipy")
        #print("\t\truntime ", time.time() - start)
        dfnc = data['FNCdyn'].astype('float32')
        return torch.from_numpy(dfnc), filepath, torch.from_numpy(self.age[k])


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import pandas as pd
    import tqdm
    rows = []
    test_dataset = ConvertData(N_subs=15885)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for batch_i, (fnc, fname, age) in tqdm.tqdm(enumerate(test_dataloader)):
        #print(fnc.shape, fname[0])
        cfname = "/data/collaboration/brainHack2022/bbaker/fnc_lstm/data/" + fname[0][1:].replace("/","-") + ".pt"
        torch.save(fnc[0], cfname)
        rows.append(dict(old_filename=fname[0], new_filename=cfname, age=age))
        #break
        df = pd.DataFrame(rows)
        df.to_csv('./data/converted_files.csv')
