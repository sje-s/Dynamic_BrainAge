from scipy.io import loadmat
import sys
import numpy as np

pred = loadmat("logs/Bag/" + sys.argv[1] + "/logs/predictions.mat") #preds
trues = loadmat("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")

trueAge = np.squeeze(trues["analysis_SCORE"][:, 0])
predAge = np.squeeze(pred["preds"])
diag = np.squeeze(trues["analysis_SCORE"][:, 2])

print("SZ Corr: " + str(np.corrcoef(predAge[np.where(diag==1)], trueAge[np.where(diag==1)])[0, 1]))
print("HC Corr: " + str(np.corrcoef(predAge[np.where(diag==2)], trueAge[np.where(diag==2)])[0, 1]))
print("SZ BAG: " + str(np.mean(predAge[np.where(diag==1)] - trueAge[np.where(diag==1)])))
print("HC BAG: " + str(np.mean(predAge[np.where(diag==2)] - trueAge[np.where(diag==2)])))
print("Corr: " + str(np.corrcoef(predAge, trueAge)[0, 1]))
print("BAG: " + str(np.mean(predAge - trueAge)))