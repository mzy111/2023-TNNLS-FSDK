function [Y] = n2nc(Y0,c)
n = size(Y0,1);      % number of samples
Y = zeros(n,c);      % binary cluster matrix
for i = 1:n
    Y(i,Y0(i)) = 1;  % binary construction
end