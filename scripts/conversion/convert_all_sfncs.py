import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
import scipy.io as sio
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
        self.filepath = np.array(all_data.loc[self.idxs, "TC_full_fname"])
        self.study_split = dict(UKB=[],HCP1_LR=[],HCP1_RL=[],HCP2_LR=[],HCP2_RL=[],HCPA=[])
        for f in self.filepath:
            if "UKB" in f:
                self.study_split['UKB'].append(f)
            elif "Aging" in f:
                self.study_split['HCPA'].append(f)
            elif 'REST1_LR' in f:
                self.study_split['HCP1_LR'].append(f)
            elif 'REST1_RL' in f:
                self.study_split['HCP1_RL'].append(f)
            elif 'REST2_LR' in f:
                self.study_split['HCP2_LR'].append(f)
            elif 'REST2_RL' in f:
                self.study_split['HCP2_RL'].append(f)
            else:
                raise(Exception("BAD FILE %s" % f))
        if 'session' in all_data.columns:
            self.session = np.array(all_data.loc[self.idxs, "session"])
        else:
            self.session = [f.split("/")[6] for f in self.filepath]
        self.cfs = None

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
        of = filepath
        dirname = os.path.dirname(filepath)
        filepath = list(glob.glob(os.path.join(dirname, "*_postprocess_results.mat")))[0]          
        cfname = "/data/users3/bbaker/projects/Dynamic_BrainAge/data/sfnc_v2/" + filepath[1:].replace("/","-") + "idx_%d.pt" % k
        #if not os.path.exists(cfname):
        data = sio.loadmat(filepath)['fnc_corrs_all']
        if of in self.study_split["UKB"]:
            idx = 0
            data = data.reshape(1,1,data.shape[0],data.shape[1])
        elif of in self.study_split["HCPA"]:
            idx = self.study_split["HCPA"].index(of)
        elif of in self.study_split['HCP1_LR']:
            idx = self.study_split['HCP1_LR'].index(of)
        elif of in self.study_split['HCP1_RL']:
            idx = self.study_split['HCP1_RL'].index(of)
        elif of in self.study_split['HCP2_LR']:
            idx = self.study_split["HCP2_LR"].index(of)
        elif of in self.study_split['HCP2_RL']:
            idx = self.study_split["HCP2_RL"].index(of)
        else:
            raise(ValueError("Bad file %s with index %d" % (of, k)))
        print("k ", k, " filepath ", filepath, " subject ", self.subID[k], " session ", self.session[k], " idx ", idx)
        data = data[:,:,TEMPLATE_IDX, :]
        data = data[:,:, :, TEMPLATE_IDX]
        data = data.squeeze()
        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        data = data[:,*np.triu_indices(53,k=1)]
        data = data[idx,...].squeeze()
        torch.save(torch.from_numpy(data), cfname)
        #return torch.from_numpy(data), filepath, torch.from_numpy(self.age[k]), self.session[k], self.subID[k]
        return dict(old_filename=filepath, new_filename=cfname, age=self.age[k][0], session=self.session[k], subject=self.subID[k])


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import pandas as pd
    import tqdm
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=0, type=int)
    args = parser.parse_args()
    rows = []
    test_dataset = ConvertData(N_subs=22569)
    get_dict = test_dataset[args.k]
    df = pd.DataFrame([get_dict])
    df.to_csv('/data/users3/bbaker/projects/Dynamic_BrainAge/data/sfnc_v2/subdirs/sub_%d.csv' % args.k, index=False)
    """
    batch_size = 1
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for batch_i, dicts in pbar:
        #print(fnc.shape, fname[0])
        dictList = [{k:dicts[k][i] for k in dicts.keys()} for i in range(batch_size)]
        rows.extend(dictList)
        #cfname = "/data/users3/bbaker/projects/Dynamic_BrainAge/data/sfnc_v2/" + fname[0][1:].replace("/","-") + ".pt"
        #if fnc != -1 and not fnc.sum() == 0.:
            #if not os.path.exists(cfname):
        #    torch.save(fnc, cfname)
        #rows.append(dict(old_filename=fname[0], new_filename=cfname, age=age.item(), session=session[0], subject=subject[0]))        
        #break
        if batch_i % 100 == 0:
            df = pd.DataFrame(rows)
            df.to_csv('./data/sfnc_converted_files_v2.csv')
        pbar.update(1)
    """
