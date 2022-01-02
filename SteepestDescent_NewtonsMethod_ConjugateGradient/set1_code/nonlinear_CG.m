%Conjugate Gradient Method, Numerical Optimization
%Written by Owen Morehead, Oct 22 2021 
%grad_fun() = gradient of cost function in grad_fun.m
%StepLength() = Step size that satisfy Wolfe conditions

clc
clear all
close all

max_iters = 5000; %maximum number of iterations
tol = 1e-16;      %tolerance level

%--- Nonlinear CG Method (FR) ---%

n = 1000;
x_fr = .01*rand(n,1);
disp("Length of x is: " + length(x_fr))
cost_min = ones(n,1);  %global minimum of cost function:
                       % $f(x_1,...,x_n) = 
                       %\sum_{i=1}^{n-1} 10(x_i^2 - x_{i+1})^2 + (x_i -
                       %1)^2$
error_fr = zeros(n,1);
error_fr(1) = norm(x_fr-cost_min);

k = 0;
grad = grad_fun(x_fr);
p = -grad;
disp("Length of p is: " + length(p))
while norm(grad) > tol && k < max_iters
    alpha = StepLength(p,x_fr); %StepLength parameters are p,x,c1,c2,a_max
    x_fr = x_fr + alpha*p;          %x_{k+1} = x_k + alpha_k*p_k
    grad_new = grad_fun(x_fr);   %grad_{k+1}
    beta = (grad_new'*grad_new)/(grad'*grad); %beta_{k+1}
    p = -grad_new + beta*p;   %update for p_{k+1}
    grad = grad_new;          %update, grad_{k+1} for next iteration
    k = k + 1;
    error(k) = norm(x_fr-cost_min,2);
    
end
disp("FR Method: Iterations stop at k = " + k)

figure; subplot(3,1,1)
grid on; hold on;
plot(1:k,log10(error(1:k)),'.-');
xlabel('k'); ylabel('Log10(Error)');
title('Nonlinear Conjugate Gradient Methods: Initial Condition: x = 0.2*rand(900,1)','fontsize',14);
%subtitle('Initial Condition: x = rand(900,1)');
legend({'FR Method'},'fontsize',15)

%--- Nonlinear CG Method with Restart (FR) ---%

x_frrest = x_fr; %use same x_fr random initial vector for all algorithms

error_frrest = zeros(n,1);
error_frrest(1) = norm(x_frrest-cost_min);

k = 0;
grad = grad_fun(x_frrest);
p = -grad;

while norm(grad) > tol && k < max_iters
    alpha = StepLength(p,x_frrest); %StepLength parameters are p,x,c1,c2,a_max
    x_frrest = x_frrest + alpha*p;          %x_{k+1} = x_k + alpha_k*p_k
    grad_new = grad_fun(x_frrest);   %grad_{k+1}
    
    %if / else statment below is the 'restart' modification 
    if abs(grad_new'*grad)/(grad_new'*grad_new) > 0.1
        beta = 0;
    else
        beta = (grad_new'*grad_new)/(grad'*grad);
    end
    
    p = -grad_new + beta*p;   %update for p_{k+1}
    grad = grad_new;          %update, grad_{k+1} for next iteration
    k = k + 1;
    error(k) = norm(x_frrest-cost_min,2);
    
end
disp("FR Method w/ Restart : Iterations stop at k = " + k)

subplot(3,1,2)
plot(1:k,log10(error(1:k)),'.-');
xlabel('k'); ylabel('Log10(Error)');
legend({'FR Method With Restart'},'fontsize',15)


%--- PR (Polka-Ribiere) Varient of Nonlinear CG Method ---%

x_pr = x_fr;
error_pr = zeros(n,1);
error_pr(1) = norm(x_pr-cost_min);

k = 0;
grad = grad_fun(x_pr);
p = -grad;

while norm(grad) > tol && k < max_iters
    alpha = StepLength(p,x_pr); %StepLength parameters are p,x,c1,c2,a_max
    x_pr = x_pr + alpha*p;          %x_{k+1} = x_k + alpha_k*p_k
    grad_new = grad_fun(x_pr);   %grad_{k+1}
    beta = (grad_new'*(grad_new - grad)/(grad'*grad)); %beta_{k+1}
    p = -grad_new + beta*p;   %update for p_{k+1}
    grad = grad_new;          %update, grad_{k+1} for next iteration
    k = k + 1;
    error(k) = norm(x_pr-cost_min,2);
    
end
disp("FR Method w/ PR Modification : Iterations stop at k = " + k)

subplot(3,1,3)
plot(1:k,log10(error(1:k)),'.-');
xlabel('k'); ylabel('Log10(Error)');
legend({'PR Method'},'fontsize',15)
           