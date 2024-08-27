from scipy.io import loadmat, savemat
import sys
import numpy as np

for i in range(5):
    for j in range(3):
        if (j == 0):
            temp = str(i)
        else:
            temp = str(i) + "_" + str(j)
        pred = loadmat("logs/Bag/Mods/Inference_Example_M_" + temp + "/logs/predictions.mat") #preds
        trues = loadmat("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")

        trueAge = np.squeeze(trues["analysis_SCORE"][:, 0])
        predAge = np.squeeze(pred["preds"])
        diag = np.squeeze(trues["analysis_SCORE"][:, 2])

        savemat("graphics/" + str(i) + "_" + str(j) + ".mat", {"hc_pred": predAge[np.where(diag==2)], "hc_true": trueAge[np.where(diag==2)], "sz_pred": predAge[np.where(diag==1)], "sz_true": trueAge[np.where(diag==1)]})

# print("SZ Corr: " + str(np.corrcoef(predAge[np.where(diag==1)], trueAge[np.where(diag==1)])[0, 1]))
# print("HC Corr: " + str(np.corrcoef(predAge[np.where(diag==2)], trueAge[np.where(diag==2)])[0, 1]))
# print("SZ BAG: " + str(np.mean(predAge[np.where(diag==1)] - trueAge[np.where(diag==1)])))
# print("HC BAG: " + str(np.mean(predAge[np.where(diag==2)] - trueAge[np.where(diag==2)])))
# print("Corr: " + str(np.corrcoef(predAge, trueAge)[0, 1]))
# print("BAG: " + str(np.mean(predAge - trueAge)))