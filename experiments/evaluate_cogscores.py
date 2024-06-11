import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.multitest import fdrcorrection as fdr
import sys
import argparse
parser = argparse.ArgumentParser

# seed that was used during randomization of data in dataloader
sorting_seed = 319

#seed that was used during kfold CV
kfold_seed = 314159

# sizes of training and validation sets
train_data_size = 10000
test_data_size = 5885

# sort phenotype data to match dataloader randomization
cog_data = pd.read_csv("/data/users2/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_cogScores2.csv")
np.random.seed(sorting_seed)
idxs = np.random.permutation(len(cog_data))

cog_data_sort = cog_data.loc[idxs]
cog_data_sort = cog_data_sort.reset_index(drop=True)

#### Test Set Results (N = 5885)
test_data = cog_data_sort.loc[train_data_size:]
for k in np.arange(1):
    test_data["PBA_run_%s"%k] = np.array(pd.read_csv('/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/logs/E0003_ukbhcp1200valid_inference/run_0/predictions/predictions.txt'%k, header = None))

test_data["BA_delta"] = test_data["PBA_run_0"] - test_data["age"]

# bias correction
from sklearn.linear_model import LinearRegression
X = np.array(test_data["age"]).reshape(-1,1)
y = np.array(test_data["PBA_run_0"])
reg = LinearRegression().fit(X, y)
alpha = reg.coef_[0]
beta = reg.intercept_

# Beheshti et al. (2019), de Lange et al. (2019b)
test_data["PBA_corrected1"] = test_data["PBA_run_0"] + (test_data["age"] - (alpha*test_data["age"] + beta))
test_data["BA_delta_corrected1"] = test_data["PBA_corrected1"] - test_data["age"]

# Cole et al. (2018)
test_data["PBA_corrected2"] = (test_data["PBA_run_0"] - beta)/alpha 
test_data["BA_delta_corrected2"] = test_data["PBA_corrected2"] - test_data["age"]

pbacorr = test_data.corr().loc["age","PBA_corrected1"]
print(f"Corr(PBA_corrected1, age): {pbacorr:.3f}")

mae = mean_absolute_error(test_data["age"], test_data["PBA_run_0"])
print(f"PBA MAE: {mae:.3f}")

mae = mean_absolute_error(test_data["age"], test_data["PBA_corrected1"])
print(f"PBA_corrected1 MAE: {mae:.3f}")

test_data.to_csv("/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/experiments/testSet_phenotypes_predictions.csv", index = False)

fields = [6,7]+list(np.arange(8,33,2))+[33,35,37]

cogLM_results = pd.DataFrame(test_data.columns[fields],columns = ['Measure'])
cogLM_results["pVal"] = 0; cogLM_results["tVal"] = 0; cogLM_results["coef"] = 0
cogLM_results["R2"] = 0; cogLM_results["R2adj"] = 0; cogLM_results["Fstat"] = 0; cogLM_results["Fstat_pVal"] = 0

for i, field in enumerate(fields):
    meas = cogLM_results.loc[i,"Measure"]
    results = smf.ols('%s ~ age + BA_delta_corrected1 + sex'%meas, data = test_data).fit()
    cogLM_results.loc[i, "pVal"] = results.pvalues['BA_delta_corrected1']
    cogLM_results.loc[i, "tVal"] = results.tvalues['BA_delta_corrected1']
    cogLM_results.loc[i, "coef"] = results.params['BA_delta_corrected1']
    cogLM_results.loc[i, "R2"] = results.rsquared
    cogLM_results.loc[i, "R2adj"] = results.rsquared_adj
    cogLM_results.loc[i, "Fstat"] = results.fvalue
    cogLM_results.loc[i, "Fstat_pVal"] = results.f_pvalue


rejected, pVal_adj = fdr(cogLM_results.pVal)

cogLM_results["significant"] = rejected
cogLM_results["pVal_FDR"] = pVal_adj

cogLM_results.to_csv("/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/experiments/testSet_BAdelta_cogMeasures_LMresults.csv", index = False)

"""
#### Training CV results

# get subjects in each fold
np.random.seed(kfold_seed)
kfold = KFold(n_splits = 10, shuffle=True, random_state = kfold_seed)

for k, (train_idx, valid_idx) in enumerate(kfold.split(np.arange(train_data_size))):
    fold_data = cog_data_sort.loc[valid_idx,:]
    predictions = pd.read_csv("/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/logs/E0004_kfold_inference/run_%s/predictions/predictions.txt"%k, header=None)
    fold_data["PBA"] = np.array(predictions)
    fold_data["BA_delta"] = fold_data["PBA"] - fold_data["age"]
    if k == 0:
        validation_data = fold_data
    else:
        validation_data = pd.concat([validation_data, fold_data])

X = np.array(validation_data["age"]).reshape(-1,1)
y = np.array(validation_data["PBA"])
reg = LinearRegression().fit(X, y)
alpha = reg.coef_[0]
beta = reg.intercept_

# Beheshti et al. (2019), de Lange et al. (2019b)
validation_data["PBA_corrected1"] = validation_data["PBA"] + (validation_data["age"] - (alpha*validation_data["age"] + beta))
validation_data["BA_delta_corrected1"] = validation_data["PBA_corrected1"] - validation_data["age"]

# Cole et al. (2018)
validation_data["PBA_corrected2"] = (validation_data["PBA"] - beta)/alpha 
validation_data["BA_delta_corrected2"] = validation_data["PBA_corrected2"] - validation_data["age"]

validation_data.to_csv("/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/experiments/validationSet_phenotypes_predictions.csv", index = False)


pbacorr = validation_data.corr().loc["age","PBA_corrected1"]
print(f"Corr(PBA_corrected1, age): {pbacorr:.3f}")

mae = mean_absolute_error(validation_data["age"], validation_data["PBA"])
print(f"PBA MAE: {mae:.3f}")

mae = mean_absolute_error(validation_data["age"], validation_data["PBA_corrected1"])
print(f"PBA_corrected1 MAE: {mae:.3f}")


cogLM_results_val = pd.DataFrame(validation_data.columns[fields],columns = ['Measure'])
cogLM_results_val["pVal"] = 0; cogLM_results_val["tVal"] = 0; cogLM_results_val["coef"] = 0
cogLM_results_val["R2"] = 0; cogLM_results_val["R2adj"] = 0; cogLM_results_val["Fstat"] = 0; cogLM_results_val["Fstat_pVal"] = 0

for i, field in enumerate(fields):
    meas = cogLM_results_val.loc[i,"Measure"]
    results = smf.ols('%s ~ age + BA_delta_corrected1 + sex'%meas, data = validation_data).fit()
    cogLM_results_val.loc[i, "pVal"] = results.pvalues['BA_delta_corrected1']
    cogLM_results_val.loc[i, "tVal"] = results.tvalues['BA_delta_corrected1']
    cogLM_results_val.loc[i, "coef"] = results.params['BA_delta_corrected1']
    cogLM_results_val.loc[i, "R2"] = results.rsquared
    cogLM_results_val.loc[i, "R2adj"] = results.rsquared_adj
    cogLM_results_val.loc[i, "Fstat"] = results.fvalue
    cogLM_results_val.loc[i, "Fstat_pVal"] = results.f_pvalue


rejected, pVal_adj = fdr(cogLM_results_val.pVal)

cogLM_results_val["significant"] = rejected
cogLM_results_val["pVal_FDR"] = pVal_adj

cogLM_results_val.to_csv("/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/experiments/validationSet_BAdelta_cogMeasures_LMresults.csv", index = False)
"""
"""

#### CADASIL Results


print(validation_data.corr())

# cadasil inference
cadasil_data = pd.read_csv('/data/users2/bbaker/projects/cadasil_analysis/CADASIL_phenotypes_withcontrols.csv')
for k in np.arange(10):
    cadasil_data["PBA_run_%s"%k] = np.array(pd.read_csv('/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/logs/E0005_cadasil_withControl_inference/run_0/predictions/predictions.txt'%k, header = None))
cadasil_data["PBA_avg"] = np.mean(cadasil_data.iloc[:,-10:],1)

# Bias correction using the alpha and beta computed from above
# Beheshti et al. (2019), de Lange et al. (2019b)
cadasil_data["PBA_corrected1"] = cadasil_data["PBA_avg"] + (cadasil_data["age"] - (alpha*cadasil_data["age"] + beta))
cadasil_data["BA_delta_corrected1"] = cadasil_data["PBA_corrected1"] - cadasil_data["age"]

# Cole et al. (2018)
cadasil_data["PBA_corrected2"] = (cadasil_data["PBA_avg"] - beta)/alpha 
cadasil_data["BA_delta_corrected2"] = cadasil_data["PBA_corrected2"] - cadasil_data["age"]

cadasil_data.to_csv("/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/logs/E0005_cadasil_withControl_inference/cadasil_withControls_brainAgePredictions_biasCorrected.csv", index = False)

print(cadasil_data.corr())

# get subjects in each fold
np.random.seed(kfold_seed)
kfold = KFold(n_splits = 10, shuffle=True, random_state = kfold_seed)

for k, (train_idx, valid_idx) in enumerate(kfold.split(np.arange(train_data_size))):
    fold_data = cog_data_sort.loc[valid_idx,["Subject","age","fluidCog","fluidCogAdj", "UKBfluidCog"]]
    predictions = pd.read_csv("/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/logs/E0004_kfold_inference/run_%s/predictions/predictions.txt"%k, header=None)
    fold_data["PBA"] = np.array(predictions)
    fold_data["BA_delta"] = fold_data["PBA"] - fold_data["age"]
    if k == 0:
        validation_data = fold_data
    else:
        validation_data = pd.concat([validation_data, fold_data])

X = np.array(validation_data["age"]).reshape(-1,1)
y = np.array(validation_data["PBA"])
reg = LinearRegression().fit(X, y)
alpha = reg.coef_[0]
beta = reg.intercept_

# Beheshti et al. (2019), de Lange et al. (2019b)
validation_data["PBA_corrected1"] = validation_data["PBA"] + (validation_data["age"] - (alpha*validation_data["age"] + beta))
validation_data["BA_delta_corrected1"] = validation_data["PBA_corrected1"] - validation_data["age"]

# Cole et al. (2018)
validation_data["PBA_corrected2"] = (validation_data["PBA"] - beta)/alpha 
validation_data["BA_delta_corrected2"] = validation_data["PBA_corrected2"] - validation_data["age"]

print(validation_data.corr())



"""