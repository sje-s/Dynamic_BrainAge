deconf=zeros(size(sfnc,1),size(sfnc,2));
% cov=[age,sex,site,edu,inc,t_t];
cov=[age,sex,site,age.^2,age.*sex];
% cov=[age,sex,site,age.^2];
% cov=[age,age.^2];
% cov=[age];
for cov_j=1:size(sfnc,2)
warning off
X=cov;
y2=sfnc(:,cov_j);
b = regress(y2, [ones(length(X),1) X ]);
Yhat = [ones(length(X),1) X]*b;
res = y2- Yhat;
deconf(:,cov_j)=res;
end
X = deconf;