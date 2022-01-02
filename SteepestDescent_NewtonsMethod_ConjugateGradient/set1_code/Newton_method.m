%Newton's Method, Numerical Optimization Method
%Written by Owen Morehead, Oct 21 2021
%cost_fun() = cost function from cost_fun.m
%grad_fun() = gradient of cost function in grad_fun.m
%hessian_fun() = hessian of cost function in hessian_fun.m
%StepLength() = Step size that satisfy Wolfe conditions

clc
clear all
close all

%--- Newton's Method ---%

max_iters = 500;  %maximum number of iterations
tol = 1e-16;      %tolerance level
dim = 3;          %dimensions of x

cost_min = [0,0,0];    %global minimizer of cost function
                       %written as 3x1 matrix = x*' (x_min transpose)
                
x_initial = rand(3,1);  %initial value of x
x_Newt = zeros(dim,max_iters);
x_Newt(:,1) = x_initial;
error_Newt = zeros(1,max_iters); 
error_Newt(1) = norm(x_Newt(:,1)-cost_min,2);

grad = grad_fun(x_Newt(:,1));
hes = hessian_fun(x_Newt(:,1));
p_Newt = -hes\grad; %\ equivalent to solving the system hes*dx = -grad

k = 1;
while k < max_iters && norm(grad,inf)>tol
    x_Newt(:,k+1) = x_Newt(:,k) + p_Newt;
    hes = hessian_fun(x_Newt(:,k+1));
    grad = grad_fun(x_Newt(:,k+1));
    p_Newt = -hes\grad;
    error_Newt(k+1) = norm(x_Newt(:,k+1),2);
    k = k+1;
end

disp(grad)

plot(1:k,log10(error_Newt(1:k)),'.');
xlabel('k'); ylabel('Log10(Error)');
title('Newtons Iteration, HW2 Problem 2','fontsize',15);
legend({'Newtons Iteration'},'fontsize',15)

for i = 5:k-1
    slope = log10(error_Newt(i)) - log10(error_Newt(i+1));
    fprintf('slope between iterations = %.3f \n',slope);
    
end
 
