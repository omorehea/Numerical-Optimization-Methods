%Steepest Descent and Newton's Method, Numerical Optimization Methods
%Written by Owen Morehead, Oct 21 2021
%cost_fun() = cost function from cost_fun.m
%grad_fun() = gradient of cost function in grad_fun.m
%hessian_fun() = hessian of cost function in hessian_fun.m
%StepLength() = Step size that satisfy Wolfe conditions

clc
clear all
close all

%--- Steepest Descent Method ---%

max_iters = 500; %maximum number of iterations
tol = 1e-16;      %tolerance level
dim = 2;          %dimensions of x
 
x_sd = zeros(dim,max_iters);    %steepest descent solution at each iteration
x_sd_errors = zeros(1,max_iters); %exact x value from cost function at each iteration

x_initial = [-10;-10];  %initial value of x
cost_min = [-1.7692923;-1.7692923];    %global minimizer of cost function
                         %written as 2x1 matrix = x*' (x_min transpose)
                
x_sd(:,1) = x_initial;
x_sd_errors(1) = norm(x_sd(:,1)-cost_min);
p = -grad_fun(x_sd(:,1));

k = 1;
while k < max_iters && norm(p) > tol  %Iterations will terminate when 
                                              %either k = max_iters, or 
                                              %p <= tolerance level
    a = StepLength(p,x_sd(:,k));   %computing step length along p at each iteration
    x_sd(:,k+1) = x_sd(:,k) + a*p;  %line search iteration method
    x_sd_errors(k+1) = norm(x_sd(:,k+1)-cost_min); %error in x between exact solution and numerical algorithm
    p = -grad_fun(x_sd(:,k+1));    %value of p for next iteration step, k+1
    k = k+1;

end

figure; subplot(2,1,1)
grid on; hold on;
plot(1:k,log10(x_sd_errors(1:k)),'.');
xlabel('k'); ylabel('Log10(Error)');
title('HW2 Problem 1: x_{initial} = [-10,-10]^T','fontsize',15);
legend({'Steepest Descent'},'fontsize',15)

%--- Newton's Method ---%

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
    error_Newt(k+1) = norm(x_Newt(:,k+1)-cost_min,2);
    k = k+1;
end

subplot(2,1,2)
plot(1:k,log10(error_Newt(1:k)),'.-');
xlabel('k'); ylabel('Log10(Error)');
legend({'Newtons Iteration'},'fontsize',15)
 
