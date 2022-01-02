function hes = hessian_fun(x)

%defines the hessian of the cost function
% x: Rn
% hes: Rn --> R

%global c

%hes = [2,-2;-2,3*x(2)^2]; %hw2 prob 1
%hes = [3*x(1)^2, 0, 0; 0, 2, -1; 0, -1, 1]; %hw2 prob 2

%hw4 prob 2: Cost function log(x'*Q*x) where Q random, symmetric, PD matrix
 global Q
 hes = (2*Q./(1 + x'*Q*x) - (4*x'*Q^2*x)./(1+ x'*Q*x)^2);
 
 %disp('hess fun size')
 %size(hes)
end