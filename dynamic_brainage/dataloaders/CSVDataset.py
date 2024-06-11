import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import mat73
import scipy.io
import time
import copy

def get_subset(dataset, idx):
    kwargs = copy.deepcopy(dataset.kwargs)
    kwargs['idx'] = idx
    return CSVDataset(**kwargs)


class CSVDataset(Dataset):
    def __init__(self,
                 N_components=53,
                 N_timepoints=448,
                 N_subs=999999999999999999,
                 include_sessions='all',
                 age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                converted_csv="/data/users3/bbaker/projects/LSTM_BrainAge/data/converted_files_v2.csv",
                sequential=False,
                label_key="age",
                subject_key='Subject',
                session_key='session',
                filename_key='new_filename',
                classification=False,
                idx=None,
                index_by_subject=True,
                use_triu=True,
                device="cpu"
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
        self.kwargs = dict(N_components=N_components, 
        N_timepoints=N_timepoints,
        N_subs = N_subs,
        include_sessions=include_sessions,
        age_csv=age_csv,
        converted_csv=converted_csv,
        sequential=sequential,
        label_key=label_key,
        subject_key=subject_key,
        session_key=session_key,
        filename_key=filename_key,
        classification=classification,
        idx=idx,
        index_by_subject=index_by_subject)
        self.N_subjects = N_subs    
        self.N_components = N_components
        self.use_triu = use_triu    
        if idx is not None:
            self.N_subjects = min(len(idx), self.N_subjects)            
        # Compute the size of the upper-triangle in the FNC matrix
        upper_triangle_size = int(N_components*(N_components - 1)/2)
        # ts_length = N_timepoints - window_size
        # These variables are useful for properly defining the RNN used
        self.dim = upper_triangle_size
        self.seqlen = N_timepoints
        self.device = device
        # Load demographic data (small enough to keep in memory)
        self.sequential = sequential
        full_data = pd.read_csv(age_csv)
        if converted_csv:
            file_df = pd.read_csv(converted_csv)
        else:
            file_df = full_data
        full_data[subject_key] = full_data[subject_key].astype(str)
        file_df[subject_key] = file_df[subject_key].astype(str)
        unique_subjects = list(set(full_data[subject_key].unique()).intersection(
            file_df[subject_key].unique().tolist()))
        self.N_subjects = min(self.N_subjects, len(unique_subjects))
        unique_sessions = full_data[session_key].unique()
        if idx is not None and index_by_subject:
            included_subjects = [list(unique_subjects)[i] for i in idx]
        else:
            included_subjects = unique_subjects[:N_subs]
        included_sessions = unique_sessions
        if include_sessions.lower() != 'all':
            if type(include_sessions) is str:
                if ',' in include_sessions:
                    include_sessions = include_sessions.split(",")
                elif ';' in include_sessions:
                    include_sessions = include_sessions.split(";")
                else:
                    include_sessions = [include_sessions]
            included_sessions = [s for s in included_sessions if s in include_sessions]
        included_files = file_df[file_df[subject_key].isin(included_subjects)]
        self.included_files = included_files[included_files[session_key].isin(included_sessions)]
        included_df = full_data[full_data[subject_key].isin(included_subjects)]
        self.included_df = included_df[included_df[session_key].isin(included_sessions)]
        #if idx is not None and not index_by_subject:
            #self.included_files = self.included_files.iloc[idx]
            #self.included_df = self.included_df.iloc[idx]
        self.sessions = self.included_df[session_key].tolist()
        self.subjects = self.included_df[subject_key].tolist()
        self.file_paths = self.included_files[filename_key].tolist()
        self.labels = self.included_df[label_key].tolist()

    def __len__(self):
        """Returns the length of the dataset
            i.e. the Number of subjects
        """
        return len(self.included_df)

    def __getitem__(self, k):
        """Get an individual FNC matrix (flattened) and Age (integer), resolving filepath format string with index
            and using mat73 to load data
        """        
        filepath = self.file_paths[k]
        if 'spectrogram' in filepath:
            data = torch.load(filepath)
            if self.seqlen is not None:
                S = min(self.seqlen, data.shape[-1])  
                data = data[:,:,:S]
            if self.sequential:
                label = self.labels[k]
                return data, torch.full((self.seqlen,), label).to(dfnc.device), k
            return data.permute([2,1,0]).float(), self.labels[k], k
        elif os.path.splitext(filepath)[-1] == ".pt":
            #filepath = self.converted['new_filename'].iloc[k]
            data = torch.load(filepath)            
            if self.seqlen is not None and self.seqlen != -1:
                data = data.squeeze()
                S = min(self.seqlen, data.shape[0])
                dfnc = torch.zeros((self.seqlen,*data.shape[1:]))
                dfnc[:S,...] = data[:S,...].float()
            else:
                dfnc = data.float()            
            if self.sequential:
                label = self.labels[k]
                try:
                    dfnc = torch.from_numpy(dfnc).to(self.device)
                except Exception:
                    pass
                return dfnc, torch.full((self.seqlen,), label).to(dfnc.device), k
            try:
                dfnc = torch.from_numpy(dfnc).to(self.device)
            except Exception:
                pass
            return dfnc, self.labels[k], k
        else:
            try:
                data = mat73.loadmat(filepath)
            except:
                data = scipy.io.loadmat(filepath)
            dfnc = data['FNCdyn'][:self.seqlen,:].astype('float32')
            if self.sequential:
                return torch.from_numpy(dfnc), torch.full(self.seqlen, torch.from_numpy(self.label[k]).item())
            return torch.from_numpy(dfnc), torch.from_numpy(self.labels[k]), k            

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    test_dataset = CSVDataset(N_subs=99999999999, sequential=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    for batch_i, (fnc, age) in enumerate(test_dataloader):
        print("Loaded batch %d with FNC shape %s, and average age %.2f" %
              (batch_i, str(fnc.shape), age.mean().item()))
