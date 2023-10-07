function [Y,Obj_y,changed] = updateY(X,W,b,Y,c,NITR_y)
% X数据矩阵：d*n
% W投影矩阵：d*c 已固定
% Y离散簇矩阵：n*c G加权离散簇矩阵：n*c  作为初始化并进行优化
% c簇类别数
% NITR： 总迭代次数
% 嵌入了两次NITR2以及10次的内循环，NITR是M与Y轮流交替的大循环，10次是为了让单独更新Y时让Y收敛（也就是所有标签不发生变化）
%% 输入定参
if nargin < 6
    NITR_y = 20;                  % 该循环一次是针对于交替循环一次M矩阵和Y矩阵
end
[~,n] = size(X);
A = X'*W+ones(n,1)*b';    
% lambda = 1 + eps;

% P = (lambda-1)*eye(n);            % positive semi-definite


% 重新初始化Y矩阵
% [idx] = kmeans(X',c);           % 每次更新Y都用kmeans重新初始化是不合理的
% Y = n2nc(idx,c);

% M = P*G+A;                        % O(n^2*c)
M = A;

Obj_y = zeros(NITR_y+1,1);          % 每轮流交替更新一轮M和Y后记录一次目标函数值，第一次是还没迭代时的目标函数，Y还是原来初始化的Y
G = Y*(Y'*Y+eps*eye(c))^-0.5;
Obj_y(1) = trace(G'*M);           % 未迭代时得到的目标函数值
changed = zeros(NITR_y,10);
% Obj_y(1) = trace(G'*M);
%% 迭代更新
for iter1 = 1:NITR_y             % M与Y轮流交替的大循环

%     [m,g] = max(M,[],2);         %求M每行最大值及其索引
%     Y = TransformL(g,c);
    yy = sum(Y.*Y);              % y'*y  1*c  O(n*c)
    ym = sum(Y.*M);              % y'*m  1*c  O(n*c)
%     [~,idxi] = sort(m);          % 更新次序，从最大值最小的开始更新？？
    for iter2 = 1:10  
        % 如果在迭代过程中只要有一个样本的所属类别发生了变化，那么converged就会变为false
        % 那么继续循环，如果一直不收敛，达到最大迭代次数后再跳出循环，一旦某一次所有样本的所属关系都
        % 没有发生变化，说明已经收敛，那么iter2就不用再迭代了，跳出迭代，此时和矩阵M没有任何关系
        converged = true;
        for i = 1:n                    % 更新Y的每一行（对应每一个样本点） 此循环复杂度为O(n*c)
%             i = idxi(iter_n);               % M中最大值小的行先更新，最大值大的放后面
            mi = M(i,:);                    % M矩阵的第i行 1*c
%             yi = Y(i,:);                  % Y矩阵的第i行 1*c
            [~,id0] = find(Y(i,:)==1);      % 该行对应样本原来的归属类别 1*1
            for k = 1:c                     % O(c)
                if k == id0
                    incre_y(k) = ym(k)/sqrt(yy(k)) - (ym(k) - mi(k))/sqrt(yy(k)-1+eps);
                else
                    incre_y(k) = (ym(k)+mi(k))/sqrt(yy(k)+1) - ym(k)/sqrt(yy(k));
                end
            end
                
%             a = ((ym+(1-yi).*mi)./sqrt(yy + 1-yi));
%             b = ((ym-yi.*mi)./(sqrt(yy-yi)+eps)); % 增量 1*c
%             incre_y = a-b;

            [~,id] = max(incre_y);          % 该行对应样本更新后的归属类别 1*1
            if id~=id0
                converged = false;          % 说明不收敛,n个样本更新完成后还要接着对Y更新
                changed(iter1,iter2) = changed(iter1,iter2)+1; % 累积计算类别变化的样本数，行对应大循环，列对应小循环
                yi = zeros(1,c);            % 重置yi
                yi(id) = 1;                 % 为新类别标签赋值
                Y(i,:) = yi;                % 更新Y矩阵的第iter_n行
                yy(id0) = yy(id0) - 1;      % id0标签从1变成0，所以yy要减一
                yy(id)  = yy(id) + 1;       % id标签从0变成1，所以yy要加一
                ym(id0) = ym(id0) - mi(id0);% id0标签从1变成0，所以ym要减去mi对应的值
                ym(id)  = ym(id) + mi(id);  % id标签从0变成1，所以ym要加上mi对应的值
            end
        end
        if converged                        % 遍历n个样本后，false说明不收敛，继续更新Y，true说明收敛，开始下一轮更新M
            break;
        end
    end
    G = Y*(Y'*Y+eps*eye(c))^-0.5;         % 时间复杂度为O(n*c^2)+O(c^3)
%     M = P*G+A;                            % 时间复杂度为O(n^2*c)
    Obj_y(iter1+1) = trace(G'*M);           % G由Y求得，M由G和gamma,U,X求得，每一轮迭代后，U会随着W大幅度变化，因此就算Y不变，该目标函数值也会发生较大变化
%     if isnan(Obj_y(iter1+1))
%         finalY = Y;
%         finalObj=Nan;
%         break;
%     end
%     if iter1 == 1
%         maxObj=Obj_y(iter1+1);
%         finalY = Y;
%         finalObj=maxObj;
%     else
%         if ~isnan(Obj_y(iter1+1)) && Obj_y(iter1+1) >= maxObj
%             maxObj=Obj_y(iter1+1);
%             finalObj=maxObj;
%             finalY = Y;
%         end
%     end
    if iter1 > 3 && (Obj_y(iter1)-Obj_y(iter1-1))/Obj_y(iter1)<1e-10
%     if iter1 == NITR_y && (Obj_y(iter1)-Obj_y(iter1-1))/Obj_y(iter1)<1e-10
        break;
    end
%     if iter1>30 && sum(abs(Obj_y(iter1-8:iter1-4)-Obj_y(iter1-3:iter1+1)))<1e-10
%         break;
%     end
end
end
    
    
                
            
