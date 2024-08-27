from scipy.io import loadmat
import sys
import numpy as np

pred = loadmat("logs/" + sys.argv[1] + "/logs/predictions.mat") #preds
trues = loadmat("/home/users/sedwardsswart/Documents/DFNC/for_Sabrina_Age_project.mat")

trueAge = np.squeeze(trues[sys.argv[2]][:, int(sys.argv[3])])
predAge = np.squeeze(pred["preds"])

temp = np.where(~np.isnan(trueAge))
trueAge = trueAge[temp]
temp = np.where(trueAge>0)
trueAge = trueAge[temp]

print("Corr: " + str(np.corrcoef(predAge, trueAge)[0, 1]))
print("BAG: " + str(np.mean(predAge - trueAge)))