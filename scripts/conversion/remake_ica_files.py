import numpy as np
import pandas as pd
import scipy,io
import itertools
import os

'''
Organize HCP phenodata
'''
#get subject list
x = scipy.io.loadmat('/data/qneuromark/Results/ICA/HCP/REST1_LR/HCP1Subject.mat')
HCP_ids = [x['files'][:,k][0][0][1].split("/")[7] for k in range(x['files'].shape[1])]

# map order of subject names in DFNC dir
hcp_filepattern = "/data/qneuromark/Results/DFNC/HCP/%s_dfnc_sub_%03d_sess_001_results.mat"
hcp_ses = ["REST1_LR/HCP1","REST1_RL/HCP2","REST2_LR/HCP3","REST2_RL/HCP4"]
hcp_fnames = [hcp_filepattern %(ses, k+1) for ses, k in list(itertools.product(hcp_ses, range(x['files'].shape[1])))]

hcp_sMRI_filepattern = "/data/qneuromark/Data/HCP/Data_BIDS/Raw_Data/%06d/T1w_MPR1/anat/Sm6mwc1pT1.nii.nii"
hcp_sMRI_fnames = [hcp_sMRI_filepattern %(id) for id in list(np.array(HCP_ids).astype('int64'))]

for i, fname in enumerate(hcp_sMRI_fnames):
    if not os.path.isfile(fname):
        hcp_sMRI_fnames[i] = ""

hcp_TC_filepattern = "/data/qneuromark/Results/ICA/HCP/%s_sub%03d_timecourses_ica_s1_.nii"
hcp_ses = ["REST1_LR/HCP1","REST1_RL/HCP2","REST2_LR/HCP3","REST2_RL/HCP4"]
hcp_TC_fnames = [hcp_TC_filepattern %(ses, k+1) for ses, k in list(itertools.product(hcp_ses, range(x['files'].shape[1])))]

for i, fname in enumerate(hcp_TC_fnames):
    if not os.path.isfile(fname):
        hcp_TC_fnames[i] = ""

hcp_SM_filepattern = "/data/qneuromark/Results/ICA/HCP/%s_sub%03d_component_ica_s1_.nii"
hcp_ses = ["REST1_LR/HCP1","REST1_RL/HCP2","REST2_LR/HCP3","REST2_RL/HCP4"]
hcp_SM_fnames = [hcp_SM_filepattern %(ses, k+1) for ses, k in list(itertools.product(hcp_ses, range(x['files'].shape[1])))]

# combine HCP subject ids with DFNC dir subject ids
hcp_subset = pd.DataFrame(data=np.array(HCP_ids*4).astype('int64'), columns = ["Subject"])
hcp_subset["TC_full_fname"] = hcp_TC_fnames
hcp_subset["SM_full_fname"] = hcp_SM_fnames
hcp_subset["DFNC_full_fname"] = hcp_fnames
hcp_subset["sMRI_full_fname"] = np.array(hcp_sMRI_fnames*4)
hcp_subset["dataset"] = np.array(["HCP"]*len(hcp_subset))
hcp_subset["session"] = np.array(["REST1_LR"]*len(HCP_ids) + ["REST1_RL"]*len(HCP_ids) + ["REST2_LR"]*len(HCP_ids) + ["REST2_RL"]*len(HCP_ids))

#grab age data
hcp = pd.read_csv('/data/qneuromark/Data/HCP/Data_info/RESTRICTED_12_2_2020_5_36_9.csv')
hcp_age = hcp.iloc[:,:2]
hcp_age = hcp_age.rename(columns={"Age_in_Yrs":"age"})

hcp_fields = pd.read_csv("/data/users3/mduda/scripts/brainAge/hcp_vars_of_interest.csv")

# get cognitive scores
hcp_data = pd.read_csv("/data/qneuromark/Data/HCP/Data_info/HCP_demo.csv")
# hcp_cog = hcp_data.loc[:,["Subject", "CogFluidComp_Unadj", "CogFluidComp_AgeAdj"]]
# hcp_cog = hcp_cog.rename(columns = {"CogFluidComp_Unadj": "fluidCog", "CogFluidComp_AgeAdj": "fluidCogAdj"})
hcp_cog = hcp_data.loc[:,list(hcp_fields.Field)]
hcp_cog = hcp_cog.rename(columns = {list(hcp_fields.Field)[i]: list(hcp_fields.Description)[i] for i in range(1,len(hcp_fields))})
hcp_cog = hcp_cog.replace('999', np.nan)
#hcp_cog["fluidCog_scaled"] = np.array((hcp_cog["fluidCog"].astype('float64') - np.min(hcp_cog["fluidCog"].astype('float64')))/(np.max(hcp_cog["fluidCog"].astype('float64')) - np.min(hcp_cog["fluidCog"].astype('float64'))))
   

# join id map with age data
hcp_subset_v1 = pd.merge(hcp_subset, hcp_age, how = "inner", on = "Subject")
hcp_subset_final = pd.merge(hcp_subset_v1, hcp_cog, how = "inner", on = "Subject")

'''
Organize HCP Aging phenodata
'''
# get subject list
y = scipy.io.loadmat('/data/qneuromark/Results/ICA/HCP_Aging/HCP_AgingSubject.mat')
HCPa_ids = [y['files'][:,k][0][0][1].split("/")[7].split("_")[0] for k in range(y['files'].shape[1])]
HCPa_ses = [y['files'][:,k][0][0][1].split("/")[9] for k in range(y['files'].shape[1])]

# map order of subject names to DFNC dir
hcpa_filepattern = "/data/qneuromark/Results/DFNC/HCP_Aging/HCP_Aging_dfnc_sub_%03d_sess_001_results.mat"
hcpa_fnames = [hcpa_filepattern %(k) for k in range(y['files'].shape[1])]

hcpa_sMRI_filepattern = "/data/qneuromark/Data/HCP_Aging/Data_BIDS/Raw_Data/%s/ses_01/anat/Sm6mwc1pT1.nii"
hcpa_sMRI_fnames = [hcpa_sMRI_filepattern %(id) for id in HCPa_ids]

hcpa_TC_filepattern = "/data/qneuromark/Results/ICA/HCP_Aging/HCP_Aging_sub%03d_timecourses_ica_s1_.nii"
hcpa_TC_fnames = [hcpa_TC_filepattern %(k) for k in range(y['files'].shape[1])]

hcpa_SM_filepattern = "/data/qneuromark/Results/ICA/HCP_Aging/HCP_Aging_sub%03d_component_ica_s1_.nii"
hcpa_SM_fnames = [hcpa_SM_filepattern %(k) for k in range(y['files'].shape[1])]


for i, fname in enumerate(hcpa_sMRI_fnames):
    if not os.path.isfile(fname):
        hcpa_sMRI_fnames[i] = ""

# combine HCP subject ids with DFNC dir ids
hcpa_subset = pd.DataFrame(data=HCPa_ids, columns = ["Subject"])
hcpa_subset["DFNC_full_fname"] = hcpa_fnames
hcpa_subset["sMRI_full_fname"] = hcpa_sMRI_fnames
hcpa_subset["TC_full_fname"] = hcpa_TC_fnames
hcpa_subset["SM_full_fname"] = hcpa_SM_fnames
hcpa_subset["dataset"] = np.array(["HCP_Aging"]*len(hcpa_subset))
hcpa_subset["session"] = np.array(HCPa_ses)


# grab age data
hcpa = pd.read_csv('/data/qneuromark/Data/HCP_Aging/Data_info/HCA_LS_2.0_subject_completeness.csv')
hcpa_age = hcpa.loc[1:,["src_subject_id","interview_age"]]
hcpa_age["age"] = np.array(hcpa_age["interview_age"]).astype('int64')/12
hcpa_age = hcpa_age.rename(columns = {"src_subject_id": "Subject"})
hcpa_age = hcpa_age.drop("interview_age", axis = 1)

hcpa_fields = pd.read_csv("/data/users3/mduda/scripts/brainAge/hcp_aging_vars_of_interest.csv")

# get cognitive scores
for i, filename in enumerate(np.unique(list(hcpa_fields.File))):
    hcpa_data = pd.read_table("/data/qneuromark/Data/HCP_Aging/Download/%s"%filename)
    hcpa_temp = hcpa_data.loc[1:,list(["src_subject_id"]+list(hcpa_fields.loc[hcpa_fields.File == filename, "Field"]))]
    if i == 0:
        hcpa_cog = hcpa_temp
    elif filename == 'tlbx_motor01.txt':
        hcpa_temp = hcpa_temp.groupby(hcpa_temp.src_subject_id).sum()
    else:
        hcpa_cog = pd.merge(hcpa_cog, hcpa_temp, how = "outer", on = "src_subject_id")


colnames = {list(hcpa_fields.Field)[i]: list(hcpa_fields.Description)[i] for i in range(len(hcp_fields))}
colnames["src_subject_id"] = "Subject"
hcpa_cog = hcpa_cog.rename(columns = colnames)
hcpa_cog = hcpa_cog.replace('999', np.nan)
# hcpa_cog["MotorDext"] = np.mean(hcpa_cog.loc[:,["MotorDextDom","MotorDextDom"]])
#hcpa_cog["fluidCog_scaled"] = np.array((hcpa_cog["fluidCog"].astype('float64') - np.min(hcpa_cog["fluidCog"].astype('float64')))/(np.max(hcpa_cog["fluidCog"].astype('float64')) - np.min(hcpa_cog["fluidCog"].astype('float64'))))
                        

# join id map with age data
hcpa_subset_v1 = pd.merge(hcpa_subset, hcpa_age, how = "inner", on = "Subject")
hcpa_subset_final = pd.merge(hcpa_subset_v1, hcpa_cog, how = "left", on = "Subject")


'''
Organize UKB phenodata --- UPDATED 04/26/2024 ----
'''
# grap age data
age_csv="/data/users3/mduda/scripts/brainAge/UKB_data_unaffected_v2.csv"
UKB_demo = pd.read_csv(age_csv)

# map filepaths
ukb_filepattern = "/data/qneuromark/Results/DFNC/UKB/%s/UKB_dfnc_sub_001_sess_001_results.mat"
ukb_fnames = [ukb_filepattern %(UKB_demo.iloc[k].SubID) for k in range(UKB_demo.shape[0])]
UKB_demo["DFNC_full_fname"] = ukb_fnames

ukb_sMRI_filepattern = "/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data/%03d/ses_01/anat/Sm6mwc1pT1.nii.nii"
ukb_sMRI_fnames = [ukb_sMRI_filepattern %(id) for id in UKB_demo.eid]

ukb_TC_filepattern = "/data/qneuromark/Results/ICA/UKBioBank/%s/UKB_sub01_timecourses_ica_s1_.nii"
ukb_TC_fnames = [ukb_TC_filepattern %(UKB_demo.iloc[k].SubID) for k in range(UKB_demo.shape[0])]

for i, fname in enumerate(ukb_TC_fnames):
    if not os.path.isfile(fname):
        pass

ukb_SM_filepattern = "/data/qneuromark/Results/ICA/UKBioBank/%s/UKB_sub01_component_ica_s1_.nii"
ukb_SM_fnames = [ukb_SM_filepattern %(UKB_demo.iloc[k].SubID) for k in range(UKB_demo.shape[0])]

for i, fname in enumerate(ukb_sMRI_fnames):
    if not os.path.isfile(fname):
        ukb_sMRI_fnames[i] = ""

UKB_demo = UKB_demo.rename(columns={'scanAge':'age'})
UKB_demo["sMRI_full_fname"] = ukb_sMRI_fnames
UKB_demo["TC_full_fname"] = ukb_TC_fnames
UKB_demo["SM_full_fname"] = ukb_SM_fnames

# get cognitive measures
ukb_cog = pd.read_csv("/data/users3/mduda/scripts/brainAge/UKB_fluidCog_scores.csv")
ukb_cog = ukb_cog.rename(columns = {"eid": "Subject", "fluid_intelligence_score_f20016_0_0": "UKBfluidCog"})
#ukb_cog["fluidCog_scaled"] = np.array((ukb_cog["fluidCog"].astype('float64') - np.min(ukb_cog["fluidCog"].astype('float64')))/(np.max(ukb_cog["fluidCog"].astype('float64')) - np.min(ukb_cog["fluidCog"].astype('float64'))))



#clean up
UKB_subset_v1 = UKB_demo.rename(columns={"eid":"Subject"})
UKB_subset_v1 = UKB_subset_v1.loc[:,["Subject","DFNC_full_fname","sMRI_full_fname","age", "session","TC_full_fname","SM_full_fname"]]
UKB_subset_final = pd.merge(UKB_subset_v1, ukb_cog, how = "left", on = "Subject")
UKB_subset_final["dataset"] = np.array(["UKB"]*len(UKB_subset_final))


'''
combine
'''
# combine 3 data sources
all_data = pd.concat([hcp_subset_final, hcpa_subset_final, UKB_subset_final])

# make sure all files exist
test_files = [os.path.isfile(path) for path in all_data.DFNC_full_fname]
all_data_final = all_data.loc[test_files]
all_data_final = all_data_final.reset_index(drop=True)

# write out csv
all_data_final.to_csv("/data/users3/bbaker/projects/Dynamic_BrainAge/scripts/conversion/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv", index = False)

