addpath(genpath('/trdapps/linux-x86_64/matlab/toolboxes/GroupICAT_latestBeta/'));
load /data/users3/bbaker/projects/pydfnc/data/fbirnp3_rest_C100_ica_TC_scrubbed_filt_RSN.mat
X = TC_rsn_filt;
corrInfo = icatb_compute_dfnc(X,2);