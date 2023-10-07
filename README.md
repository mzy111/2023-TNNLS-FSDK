# FSDK

Using the code, please cite:
Nie F, Ma Z, Wang J, et al. Fast Sparse Discriminative K-Means for Unsupervised Feature Selection[J]. IEEE Transactions on Neural Networks and Learning Systems, early access, Jan 25, 2023, doi: 10.1109/TNNLS.2023.3238103.

Paper URL: https://ieeexplore.ieee.org/document/10026244

The code explanation: 
The main function of the code: FSDK.m
The clustering validation function: Kmeans_validate.m
You can use run.m to perform FSDK for USPS data set.
If you have any questions, please connect zhenyu.ma@mail.nwpu.edu.cn

# Use of Main Function

FS(feature selection): [X_select,Obj_w,idxw,fea_id,t,converge] = FSDK(X,label,s_num,gamma,p,NITR_w,NITR_y)

n: the number of instances in primal data; d: the number of dimensions in primal data;

Input:

X: primal data matrix; label: (n \times 1) true label vector; s_num: the number of selected features; gamma: regularized parameter of L2,p-norm; p: p value of L2,p-norm (0,1].

Output:

X_select: data matrix after FS; Obj_w: change of main objective function; idxw: ranking of all features according to W; fea_id: index of selected features; t: running time of FS; converge: 1 means converge and 0 means not converge.

example on USPS data set with 110 selected features:

load('USPS_data.mat')
[X_select,Obj_w,idxw,fea_id,t,converge] = FSDK(X,label,110,10,1); % gamma=10 and p=1 according to paper
[result_end] = Kmeans_validate(X_select,label);


