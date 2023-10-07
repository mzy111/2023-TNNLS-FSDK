function [X_select,Obj_w,idxw,fea_id,t,converge] = FSDK(X,label,s_num,gamma,p,NITR_w,NITR_y)

% X：原始数据矩阵 n*d

% label：真实类别标签 n*1

% s_num：特征选择数量

% p：2,p范数范数参数（0,2]

% gamma：2,p范数的正则化参数；
       % gamma越大，越接近F范数，W行稀疏性越弱；p>=1时为凸
       % gamma越小，越接近2,0范数，W行稀疏性越强。0<p<1时为非凸
       
% NITR_w——更新W与b的总迭代次数

% NITR_y——更新离散Y的迭代次数

%% 参数输入

if nargin < 7
    NITR_y = 20; % 内环最大迭代次数30次
end
if nargin < 6
    NITR_w = 30; % 外环最大迭代次数30次
end
if nargin < 5
    p = 1;       % 不输入默认2,1范数
end
if nargin < 4
    gamma = 100;
end
if nargin == 2
    error('请在第三个输入项输入选择的特征数量')
end
if nargin < 2
    error('请输入原始数据和真实标签')
end

%% 输入 构造中心矩阵的时间复杂度是O(nd)
X = X';
[d,n] = size(X);
% H = eye(n)-ones(n)./n;
X_mean = mean(X,2);          % 时间复杂度O(nd)
X = X - X_mean;              % 原始数据中心化 时间复杂度O(nd)
% X = X * H;
c = length(unique(label));   % 真实聚类簇数
converge = true;             % 收敛逻辑值

%% 初始化 rand初始化的时间复杂度是O(nc)
err_objw = 1;                % 目标函数差
iter = 1;                    % while中迭代次数
U = eye(d);                  % 初始化对角阵U，与2,p范数相关
pause(0)
tic
init_Y = 'rand';             % Y矩阵的初始化方式
switch init_Y
    case 'rand'              % Y矩阵随机初始化
        Y = zeros(n,c);
        for i = 1:n
            Y(i,randperm(c,1)) = 1;
        end
    case 'kmeans'                                            % 利用K-means初始化
        [idx] = kmeans(X',c);
        Y = n2nc(idx,c);
end
G = Y*(Y'*Y+eps*eye(c))^(-0.5);                              % 时间复杂度O(n*c^2)+O(c^3)


% Clus_resultY = zeros(NITR_w+1,3);
% labelpre = labelconvert(Y);
% Clus_resultY(1,:) = ClusteringMeasure(labelpre,label);
%% Fix Y and update W,b
while iter<= NITR_w
    % 固定Y，更新W （Algorithm 1）
    b = (G'*ones(n,1))./n;                                   % 更新b 时间复杂度O(n*c)+O(c)
    WUiter = 'Once';
    switch WUiter
        case 'Once'
%             P = X'*(X*X'+ gamma*U + eps.*eye(d))^(-1)*X+(1/n)*ones(n);
%             a = svd(P); e = rank(P);
            W = (X*X'+ gamma*U + eps.*eye(d))^(-1)*X*G;                  % 更新W，只更新一次
            twopnormw = zeros(d,1);
            for j = 1:d
                U(j,j) = (0.5*p)/((norm(W(j,:),2)^2)^(1-0.5*p)+eps);
                twopnormw(j) = norm(W(j,:),2)^p;                         % 按W更新U
            end
        case 'NITR'
            uw = 10;
            for u = 1:uw
                W = (X*X'+ gamma*U + eps.*eye(d))^(-1)*X*G;              % 更新W 时间复杂度O(n*d^2)+O(d^3)+O(n*c*d)
                twopnormw = zeros(d,1);
                for j = 1:d                                              % 该内循环的时间复杂度为O(cd)
                    U(j,j) = (0.5*p)/((norm(W(j,:),2)^2)^(1-0.5*p)+eps); % 更新对角矩阵U
                    twopnormw(j) = norm(W(j,:),2)^p;
                end
            end
    end
    
    
    
    % 固定W，更新Y （Algorithm 2）
    [Y,~,~] = updateY(X,W,b,Y,c,NITR_y); % 更新Y  复杂度为NITR_y * O(n*c+n*c^2+c^3+n^2*c)+O(n^2*d+n*d^2+d^3) 复杂度...
    G = Y*(Y'*Y+eps*eye(c))^(-0.5);
    
%     labelpre = labelconvert(Y);
%     Clus_resultY(iter,:) = ClusteringMeasure(labelpre,label);
    
    
    % 外环一轮更新结束后计算目标函数值
    Obj_w(iter) = norm(X'*W + ones(n,1)*b'- G,'fro')^2 ++ gamma*sum(twopnormw); % W矩阵是主体 
    
    
    
    % 判断迭代是否收敛
    if iter>1
        err_objw = Obj_w(iter)-Obj_w(iter-1);
        if err_objw > 0
            converge = false;
        end
    end
    if iter>2 && abs(err_objw)<1e-3
        break;
    end
    
%     if iter == 1
%         Obj_y0 = Obj_y;
%         figure(1)
%         x = 1:1:size(Obj_y,1);
%         plot(x,Obj_y,'-o','MarkerSize',6,'linewidth',1.5,'Color',[0.8477 0.0156 0.1602]); % 画在第一次迭代
%         xlabel('The Number of Iterations')
%         ylabel('Objective Value')
%         grid on;
%     end
    iter = iter+1;
end


%% 特征选择
score = sum((W.*W),2);
[~,idxw] = sort(score,'descend');
t = toc;
fea_id = idxw(1:s_num);
X_select = X(fea_id,:);

%% 目标函数
% figure(2)
% x = 1:1:size(Obj_w,2);
% plot(x,Obj_w,'-o','MarkerSize',6,'linewidth',1.5,'Color',[0.4453 0.0352 0.7148]);
% xlabel('The Number of Iterations')
% ylabel('Objective Value')
% grid on;
% grid minor;