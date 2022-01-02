 function cost = cost_fun(x)

%defines the cost function
% x: Rn
% cost: Rn --> R

%global c

%cost = (c*x(1)-2)^4 + (x(2)^2)*((c*x(1)-2)^2) + (x(2)+1)^2; %hw1 prob 6
%cost = (x(1)-x(2))^2 + 1/4*(x(2))^4 - x(2)^2 + 2*x(2); %hw2 prob 1
%cost = 1/4*x(1)^4 + 1/2*(x(2)-x(3))^2 + 1/2*x(2)^2; %hw2 prob 2

%hw2 prob 6
% 
% cost = 0;
% 
% for i = 1:(length(x)-1)
%     cost = cost + (10*(x(i)^2 - x(i+1))^2 + (x(i)-1)^2);
%     
% end


%hw3 prob 2 Extended Rosenbrock function

cost = 0;
global ALPHA

for i = 1:(length(x)/2)
    cost = cost + (ALPHA*(x(2*i) - x(2*i-1)^2)^2 + (1 - x(2*i-1))^2);
    
end


