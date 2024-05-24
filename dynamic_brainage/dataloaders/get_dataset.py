from dynamic_brainage.dataloaders.fake_fnc import FakeFNC
#from dataloaders.load_dev_data import UKBData
#from dataloaders.load_UKB_HCP1200 import UKBHCP1200Data
#from dataloaders.cadasil import CadasilData
from dynamic_brainage.dataloaders.CSVDataset import CSVDataset


def get_dataset(key, *args, **kwargs):
    if key.lower() == "fakefnc":
        return FakeFNC(*args, **kwargs)
    #elif key.lower() == "ukb":
    #    return UKBData(*args, **kwargs)
    elif key.lower() == "ukbhcp":
        return CSVDataset(*args, 
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/LSTM_BrainAge/data/converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2":
        return CSVDataset(*args, 
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/LSTM_BrainAge/data/converted_files_v2.csv",
                          **kwargs)    
    elif key.lower() == "tctest":
        return CSVDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/tc_static.csv",
                          **kwargs)
    #elif key.lower() == "cadasil":
    #    return CadasilData(*args, **kwargs)
