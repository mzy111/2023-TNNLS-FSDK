% The purpose of this code is to convert the label indicator matrix into
% label vector
function [labelY] = labelconvert(Y)
[n,~] = size(Y);
labelY = zeros(n,1);
for i = 1:n
    labelY(i) = find(Y(i,:)==1);
end