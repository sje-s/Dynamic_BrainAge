from dynamic_brainage.dataloaders.fake_fnc import FakeFNC
#from dataloaders.load_dev_data import UKBData
#from dataloaders.load_UKB_HCP1200 import UKBHCP1200Data
#from dataloaders.cadasil import CadasilData
from dynamic_brainage.dataloaders.CSVDataset import CSVDataset
from dynamic_brainage.dataloaders.CSVDataset_preload import RAMDataset

def get_ramset(key, *args, **kwargs):
    if key.lower() == "fakefnc":
        return FakeFNC(*args, **kwargs)
    #elif key.lower() == "ukb":
    #    return UKBData(*args, **kwargs)
    elif key.lower() == "ukbhcp":
        return RAMDataset(*args, 
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2":
        return RAMDataset(*args, 
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v2.csv",
                          **kwargs) 
    elif key.lower() == "ukbhcp_v2_smri":
        return RAMDataset(*args, 
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/smri_converted_files_v2.csv",
                          **kwargs) 
    elif key.lower() == "ukbhcp_v2_sfnc":
        return RAMDataset(*args, 
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/sfnc_converted_files_v2.csv",
                          **kwargs) 
    elif key.lower() == "ukbhcp_v2_tc_static":
        return RAMDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/tc_static.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_spectrogram":
        return RAMDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/spectrograms_v2.csv",
                          **kwargs)
    #elif key.lower() == "cadasil":
    #    return CadasilData(*args, **kwargs)

def get_dataset(key, *args, **kwargs):
    if key.lower() == "fakefnc":
        return FakeFNC(*args, **kwargs)
    #elif key.lower() == "ukb":
    #    return UKBData(*args, **kwargs)
    elif key.lower() == "ukbhcp":
        return CSVDataset(*args, 
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2":
        return CSVDataset(*args, 
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v2.csv",
                          **kwargs) 
    elif key.lower() == "ukbhcp_v2_smri":
        return CSVDataset(*args, 
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/smri_converted_files_v2.csv",
                          **kwargs) 
    elif key.lower() == "ukbhcp_v2_sfnc":
        return CSVDataset(*args, 
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/sfnc_converted_files_v2.csv",
                          **kwargs) 
    elif key.lower() == "ukbhcp_v2_tc_static":
        return CSVDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/tc_static.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_spectrogram":
        return CSVDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/spectrograms_v2.csv",
                          **kwargs)
    #elif key.lower() == "cadasil":
    #    return CadasilData(*args, **kwargs)

if __name__=="__main__":
    dataset = get_dataset(
        "ukbhcp_v2_tc_static",
        N_subs=17522,
        sequential=False,
        N_timepoints=448
    )
    print(dataset[0][0].shape)
    print(dataset[50][0].shape)
    print(dataset[5000][0].shape)
    print(dataset[10000][0].shape)
    print(dataset[15000][0].shape)
    print(dataset[-1][0].shape)
