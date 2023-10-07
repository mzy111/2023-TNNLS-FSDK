%% Data

load('USPS_data.mat')

%% Feature Selection

gamma = 10;
p = 1;
s_num = 110;
[X_select,Obj_w,idxw,fea_id,t,converge] = FSDK(X,label,s_num,gamma,p);

%% Clustering Validation

[result_end] = Kmeans_validate(X_select,label);
% result_end第一行：重复实验聚类均值
% 第二行：重复实验聚类标准差