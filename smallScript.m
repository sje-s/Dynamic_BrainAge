%%
load("predictions.mat")
load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")

y_pred = double(preds);
y_true = analysis_SCORE(:, 1);

scatter(y_pred, y_true)
xlabel("Predictions")
ylabel("True")
%%
bag = preds - analysis_SCORE(:, 1);
%%
scatter(preds, analysis_SCORE(:, 1))
%%
diag = analysis_SCORE(:, 3);
sz = diag == 1;
hc = diag == 2;
ageTrue = analysis_SCORE(:, 1);
szTrue = ageTrue(sz);
szPreds = preds(sz);
hcTrue = ageTrue(hc);
hcPreds = preds(hc);

szBag = szPreds - szTrue;
hcBag = hcPreds - hcTrue;

mean(szBag)
mean(hcBag)
%%
clear
load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")
load("logs/Inference_Example_23_1/logs/predictions.mat")

diag = analysis_SCORE(:, 3);
sz = diag == 1;
hc = diag == 2;
ageTrue = analysis_SCORE(:, 1);
szTrue = ageTrue(sz);
szPreds = preds(sz);
hcTrue = ageTrue(hc);
hcPreds = preds(hc);

szBag = szPreds - szTrue;
hcBag = hcPreds - hcTrue;

mean(szBag)
mean(hcBag)
%%
clear
load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")

corrsAll = zeros(10, 10);
corrsHC = zeros(10, 10);
corrsSZ = zeros(10, 10);

for i=5:14
    for j=1:311
        if (analysis_SCORE(j, i) == -9999)
            analysis_SCORE(j, i) = nan;
        end
    end
end
asSZ = analysis_SCORE(analysis_SCORE(:, 3) == 1, :);
asHC = analysis_SCORE(analysis_SCORE(:, 3) == 2, :);

for i=5:14
    for j=5:14
        temp = corrcoef(analysis_SCORE(:, i), analysis_SCORE(:, j), 'Rows', 'complete');
        corrsAll(i-4, j-4) = temp(1,2);
        temp = corrcoef(asSZ(:, i), asSZ(:, j), 'Rows', 'complete');
        corrsSZ(i-4, j-4) = temp(1,2);
        temp = corrcoef(asHC(:, i), asHC(:, j), 'Rows', 'complete');
        corrsHC(i-4, j-4) = temp(1,2);
    end
end

writematrix(corrsAll, "correlations.xlsx", 'Sheet', 'All', 'Range', 'B2');
writematrix(corrsSZ, "correlations.xlsx", 'Sheet', 'SZ', 'Range', 'B2');
writematrix(corrsHC, "correlations.xlsx", 'Sheet', 'HC', 'Range', 'B2');
%%
%TODO: finish extra stats
clear
load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")

corrsAll = zeros(10, 10);
corrsHC = zeros(10, 10);
corrsSZ = zeros(10, 10);

for i=5:14
    for j=1:311
        if (analysis_SCORE(j, i) == -9999)
            analysis_SCORE(j, i) = nan;
        end
    end
end
asSZ = analysis_SCORE(analysis_SCORE(:, 3) == 1, :);
asHC = analysis_SCORE(analysis_SCORE(:, 3) == 2, :);

for i=5:14
    temp = corrcoef(analysis_SCORE(:, i), analysis_SCORE(:, 1), 'Rows', 'complete');
    corrsAll(i-4, 1) = temp(1,2);
    temp = corrcoef(asSZ(:, i), asSZ(:, 1), 'Rows', 'complete');
    corrsSZ(i-4, 1) = temp(1,2);
    temp = corrcoef(asHC(:, i), asHC(:, 1), 'Rows', 'complete');
    corrsHC(i-4, 1) = temp(1,2);
end

writematrix(corrsAll, "stats.xlsx", 'Sheet', 'All', 'Range', 'B2');
writematrix(corrsSZ, "stats.xlsx", 'Sheet', 'SZ', 'Range', 'B2');
writematrix(corrsHC, "stats.xlsx", 'Sheet', 'HC', 'Range', 'B2');
%%
clear
load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")
for i=5:14
    for j=1:311
        if (analysis_SCORE(j, i) == -9999)
            analysis_SCORE(j, i) = nan;
        end
        if ((i == 14) && (analysis_SCORE(j, i) > 1000))
            analysis_SCORE(j, i) = nan;
        end
    end
end
temp = cell(5, 1);
for i=1:5
    temp{i} = "Model " + (i - 1);
end
writecell(FILE_ID(5:14), "BAG_Corr.xlsx", "Range", "B1")
writecell(temp, "BAG_Corr.xlsx", "Range", "A2")

corrs = zeros(5, 10);
for i=0:4
% for i=0:0
    clearvars -except i corrs
    load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")
    for k=5:14
        for j=1:311
            if (analysis_SCORE(j, k) == -9999)
                analysis_SCORE(j, k) = nan;
            end
            if ((k == 14) && (analysis_SCORE(j, k) > 1000))
                analysis_SCORE(j, k) = nan;
            end
        end
    end
    load("logs/Bag/Mods/Inference_Example_M_" + i + "/logs/predictions.mat")
    
    diag = analysis_SCORE(:, 3);
    sz = diag == 1;
    % hc = diag == 2;
    ageTrue = analysis_SCORE(:, 1);
    szTrue = ageTrue(sz);
    szPreds = preds(sz);
    % hcTrue = ageTrue(hc);
    % hcPreds = preds(hc);
    
    szBag = szPreds - szTrue;
    % hcBag = hcPreds - hcTrue;

    asSZ = analysis_SCORE(analysis_SCORE(:, 3) == 1, :);

    for j=5:14
%     for j=5:5
        temp = corrcoef(asSZ(:, j), szBag, 'Rows', 'complete');
        corrs(i + 1, j-4) = temp(1,2);
%         X = [ones(length(szBag),1) szBag];
%         b = X\asSZ(:, j);
%         yCalc = X*b;
        mdl = fitlm(szBag, asSZ(:, j));
        [R, P] = corrcoef(asSZ(:, j), szBag, 'Rows', 'complete');

%         scatter(szBag, asSZ(:, j), 'o');
%         hold on
%         plot(szBag, yCalc)
        plot(mdl)
        xlabel("Brain Age Gap")
        ylabel(FILE_ID{j})
        title("Model " + i + ", R = " + round(R(1,2), 3) + ", P = " + round(P(1,2), 3))
%         legend on
        ax = gca;
        exportgraphics(ax, "graphics/bag_sym/Sym" + j + "_Model" + i + ".jpg")
    end
end
writematrix(corrs, "BAG_Corr.xlsx", "Range", "B2")
%%
clear
load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")
for i=5:14
    for j=1:311
        if (analysis_SCORE(j, i) == -9999)
            analysis_SCORE(j, i) = nan;
        end
        if ((i == 14) && (analysis_SCORE(j, i) > 1000))
            analysis_SCORE(j, i) = nan;
        end
    end
end
temp = cell(5, 1);
for i=1:5
    temp{i} = "Model " + (i - 1);
end
writecell(FILE_ID(5:14), "BAG_Corr.xlsx", "Range", "B1")
writecell(temp, "BAG_Corr.xlsx", "Range", "A2")

corrs = zeros(5, 10);
for i=0:4
    clearvars -except i corrs
    load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")
    for k=5:14
        for j=1:311
            if (analysis_SCORE(j, k) == -9999)
                analysis_SCORE(j, k) = nan;
            end
            if ((k == 14) && (analysis_SCORE(j, k) > 1000))
                analysis_SCORE(j, k) = nan;
            end
        end
    end
    load("logs/Bag/Mods/Inference_Example_M_" + i + "/logs/predictions.mat")
    
    diag = analysis_SCORE(:, 3);
    sz = diag == 1;
    ageTrue = analysis_SCORE(:, 1);
    szTrue = ageTrue(sz);
    szPreds = preds(sz);
    
    szBag = szPreds - szTrue;
    asSZ = analysis_SCORE(analysis_SCORE(:, 3) == 1, :);

    sfnc = szBag;
    age = szTrue;
    sex = asSZ(:, 2);
    site = asSZ(:, 4);
    cpz = asSZ(:, 14);
    run("regress_out_cov.m")
    szBag = X;

    for j=5:14
        mdl = fitlm(szBag, asSZ(:, j));
        [R, P] = corrcoef(asSZ(:, j), szBag, 'Rows', 'complete');

        plot(mdl)
        xlabel("Brain Age Gap")
        ylabel(FILE_ID{j})
        title("Model " + i + ", R = " + num2str(round(R(1,2), 3), '%.e') + ", P = " + num2str(round(P(1,2), 3), '%.e'))
        ax = gca;
        exportgraphics(ax, "graphics/bag_sym/Sym" + j + "_Model" + i + ".jpg")
    end
end
%%
clear
load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")
for i=5:14
    for j=1:311
        if (analysis_SCORE(j, i) == -9999)
            analysis_SCORE(j, i) = nan;
        end
%         if ((i == 14) && (analysis_SCORE(j, i) > 1000))
%             analysis_SCORE(j, i) = nan;
%         end
    end
end
temp = cell(5, 1);
for i=1:5
    temp{i} = "Model " + (i - 1);
end
writecell(FILE_ID(5:14), "BAG_Corr.xlsx", "Range", "B1")
writecell(temp, "BAG_Corr.xlsx", "Range", "A2")

corrs = zeros(5, 10);
for i=0:4
    clearvars -except i szBagAll
    load("/data/neuromark2/Results/DFNC/FBIRN/FBIRN_DFNC_only.mat")
    for k=5:14
        for j=1:311
            if (analysis_SCORE(j, k) == -9999)
                analysis_SCORE(j, k) = nan;
            end
            if ((k == 14) && (analysis_SCORE(j, k) > 1000))
                analysis_SCORE(j, k) = nan;
            end
        end
    end
    load("logs/Bag/Mods/Inference_Example_M_" + i + "/logs/predictions.mat")
    
    diag = analysis_SCORE(:, 3);
    sz = diag == 1;
    ageTrue = analysis_SCORE(:, 1);
    szTrue = ageTrue(sz);
    szPreds = preds(sz);
    
    szBag = szPreds - szTrue;
    asSZ = analysis_SCORE(analysis_SCORE(:, 3) == 1, :);

    sfnc = szBag;
    age = szTrue;
    sex = asSZ(:, 2);
    site = asSZ(:, 4);
    cpz = asSZ(:, 14);
    run("regress_out_cov.m")
    szBag = X;

    if (i == 0)
        szBagAll = szBag;
    else
        szBagAll = szBagAll + szBag;
    end
end

szBagAll = szBagAll/5;

for j=5:14
    mdl = fitlm(szBagAll, asSZ(:, j));
    [R, P] = corrcoef(asSZ(:, j), szBagAll, 'Rows', 'complete');

    plot(mdl)
    xlabel("Brain Age Gap")
    ylabel(FILE_ID{j})
    title(FILE_ID{j} + ", R = " + num2str(round(R(1,2), 3), '%.3e') + ", P = " + num2str(round(P(1,2), 3), '%.3e'))
    ax = gca;
    exportgraphics(ax, "graphics/bag_sym/Sym" + j + "_ModelAll.jpg")
end
%%
age = table2array(Demos(:, "Age"));
sex = table2array(Demos(:, "Sex"));
pred = table2array(Demos(:, "Model 0 Predictions"));
hamd = table2array(Demos(:, "HAMD"));

age1 = age(sex == 1);
age2 = age(sex == 2);
pred1 = pred(sex == 1);
pred2 = pred(sex == 2);
hamd1 = hamd(sex == 1);
hamd2 = hamd(sex == 2);

hcAge1 = age1(isnan(hamd1));
mddAge1 = age1(~isnan(hamd1));
hcPred1 = pred1(isnan(hamd1));
mddPred1 = pred1(~isnan(hamd1));
hcBag1 = hcPred1 - hcAge1;
mddBag1 = mddPred1 - mddAge1;


hcAge2 = age2(isnan(hamd2));
mddAge2 = age2(~isnan(hamd2));
hcPred2 = pred2(isnan(hamd2));
mddPred2 = pred2(~isnan(hamd2));
hcBag2 = hcPred2 - hcAge2;
mddBag2 = mddPred2 - mddAge2;

[h1, p1] = ttest2(hcBag1, mddBag1)
[h2, p2] = ttest2(hcBag2, mddBag2)