function [result_end] = Kmeans_validate(X_select,label,iter_k)

if nargin < 3
    iter_k = 20;   % kmeans次数
end

c = length(unique(label));
result_iter = zeros(iter_k,3);

for i = 1:iter_k
    [predY] = kmeans(X_select',c);
    result_iter(i,:) = ClusteringMeasure(label,predY);
end

result_mean = mean(result_iter);
result_std = std(result_iter);
result_end = zeros(2,3);
result_end(1,:) = result_mean; % 均值
result_end(2,:) = result_std;  % 标准差