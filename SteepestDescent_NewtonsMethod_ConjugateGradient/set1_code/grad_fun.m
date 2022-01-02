function grad = grad_fun(x)

%defines the gradient of the cost function
% x: Rn
% grad: Rn --> R

%global c

%grad = [4*c*(c*x(1)-2)^3 + 2*c*x(2)^2*(c*x(1)-2);2*x(2)*(c*x(1)-2)^2 + 2*(x(2)+1)]; %hw1 prob 6
%grad = [2*(x(1)-x(2));x(2)^3 - 2*x(1) + 2]; %hw2 prob 1
%grad = [x(1)^3;2*x(2)-x(3);x(3)-x(2)];  %hw2 prob 2

%hw2 prob 6
n = length(x);
grad = zeros(n,1);

grad(1) = 40*x(1)*(x(1)^2 - x(2)) + 2*(x(1)-1);
grad(n) = -20*(x(n-1)^2 - x(n));
for i = 2:n-1
    grad(i) = 40*x(i)*(x(i)^2 - x(i+1)) + 2*(x(i)-1) - 20*(x(i-1)^2 - x(i)); 

end
