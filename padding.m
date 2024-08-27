%Pad to 448
%%
clear
load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")

for i=1:311
    pad = zeros(311, 1378);
    DFNC_FBIRN{i} = cat(1, pad, DFNC_FBIRN{i});
end

save("paddedFrontFBIRN.mat", "DFNC_FBIRN")
%%
clear
load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")

for i=1:311
    pad = zeros(311, 1378);
    DFNC_FBIRN{i} = cat(1, DFNC_FBIRN{i}, pad);
end

save("paddedBackFBIRN.mat", "DFNC_FBIRN")
%%
clear
load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")

for i=1:311
    pad1 = zeros(156, 1378);
    pad2 = zeros(155, 1378);
    DFNC_FBIRN{i} = cat(1, pad1, DFNC_FBIRN{i});
    DFNC_FBIRN{i} = cat(1, DFNC_FBIRN{i}, pad2);
end

save("paddedBothFBIRN.mat", "DFNC_FBIRN")