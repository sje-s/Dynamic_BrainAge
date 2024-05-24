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


class MultiModalDataset(Dataset):
    def __init__(self,
                 N_components=53,
                 N_timepoints=448,
                 N_subs=999999999999999999,
                 include_sessions='all',
                 age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                converted_csvs=["/data/users3/bbaker/projects/LSTM_BrainAge/data/converted_files_v2.csv",
                                "/data/users3/bbaker/projects/LSTM_BrainAge/data/converted_files_v2.csv"],
                sequential=False,
                label_key="age",
                subject_key='Subject',
                session_key='session',
                filename_key='new_filename',
                classification=False,
                idx=None,
                index_by_subject=True
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
        converted_csvs=converted_csvs,
        sequential=sequential,
        label_key=label_key,
        subject_key=subject_key,
        session_key=session_key,
        filename_key=filename_key,
        classification=classification,
        idx=idx,
        index_by_subject=index_by_subject)
        self.N_subjects = N_subs        
        if idx is not None:
            self.N_subjects = min(len(idx), self.N_subjects)            
        # Compute the size of the upper-triangle in the FNC matrix
        upper_triangle_size = int(N_components*(N_components - 1)/2)
        # ts_length = N_timepoints - window_size
        # These variables are useful for properly defining the RNN used
        self.dim = upper_triangle_size
        self.seqlen = N_timepoints
        # Load demographic data (small enough to keep in memory)
        self.sequential = sequential
        full_data = pd.read_csv(age_csv)
        file_dfs = []
        if converted_csvs:
            file_dfs = [pd.read_csv(converted_csv) for converted_csv in converted_csvs]
        else:
            file_dfs = [full_data]
        full_data[subject_key] = full_data[subject_key].astype(str)
        for i in range(len(file_dfs)):
            file_dfs[i][subject_key] = file_dfs[i][subject_key].astype(str)
        unique_subjects = full_data[subject_key].unique()
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
        self.included_files = []
        for i in range(len(file_dfs)):
            inc = file_dfs[i][file_dfs[i][subject_key].isin(included_subjects)]
            self.included_files.append(inc[inc[session_key].isin(included_sessions)])
        included_df = full_data[full_data[subject_key].isin(included_subjects)]
        self.included_df = included_df[included_df[session_key].isin(included_sessions)]
        #if idx is not None and not index_by_subject:
            #self.included_files = self.included_files.iloc[idx]
            #self.included_df = self.included_df.iloc[idx]
        self.sessions = self.included_df[session_key].tolist()
        self.subjects = self.included_df[subject_key].tolist()
        self.file_paths = []
        for i in range(len(file_dfs)):
            self.file_paths.append(self.included_files[i][filename_key].tolist())
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
        filepaths = [fp[k] for fp in self.file_paths]
        datasets = []
        label = self.labels[k]
        for filepath in filepaths:
            data = torch.load(filepath)
            datasets.append(data)
        return datasets, label

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    test_dataset = MultiModalDataset(N_subs=99999999999, sequential=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    for batch_i, (fnc, age) in enumerate(test_dataloader):
        print("Loaded batch %d with FNC shape %s, and average age %.2f" %
              (batch_i, str(fnc.shape), age.mean().item()))
