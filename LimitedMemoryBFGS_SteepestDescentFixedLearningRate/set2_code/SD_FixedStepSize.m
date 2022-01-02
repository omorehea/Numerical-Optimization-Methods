%Gradient Descent Method with Fixed Learning Rate, Numerical Optimization
%Written by Owen Morehead, Nov 6 2021
%grad_fun() = gradient of cost function in grad_fun.m
%Method applied on quadratic cost function


clc
clear all
close all

global A b

max_iters = 10000; %maximum number of iterations
tol = 1e-10;      %tolerance level
n = 100;         %dimension of matrix A and vector x
         
x = zeros(n,max_iters);
error = zeros(1,max_iters);

lambda1 = 1;            %10 + epsilon*(rand(1,m)); range of lambda
lambda2 = 100;          %1000 + epsilon*(rand(1,n-m)); second range of lambda

lambda = linspace(lambda1,lambda2,n);  %sort([lambda1,lambda2]); sort the two ranges together

[Q,R] = qr(rand(n,n)); %QR factorization to create a random orthogonal matrix Q
A = Q'*diag(lambda)*Q; %matrix a is symmetric, positive def

x_g = rand(n,1);    %random guess for optimal solution
b = A*x_g;          %b = Ax^*

x0 = 2*rand(n,1); %initial random step

c = 0.015;
display(lambda)
for alpha = [c,c/2,(2/(lambda1+lambda2))] %choices for constant learning rate, a.
    x = x0;
    k = 1;
    error(k) = norm(x-x_g,2); %initial error
    while k < max_iters && error(k) > tol  %Iterations will terminate when 
                                              %either k = max_iters, or 
                                              %p <= tolerance level
    
        p = -grad_fun(x);    %value of p for next iteration step, k+1                                   
        x = x + alpha*p;  %line search iteration method
        error(k+1) = norm(x-x_g,2); %error in x between exact solution and numerical algorithm
        k = k+1;
    
    end


    figure(1);
    grid on; hold on;
    plot(1:k,log10(error(1:k)),'.-'); %numerical result
    
end

xlabel('k'); ylabel('Log10(Error)');
title(sprintf('Gradient Descent with Fixed Learning Rate'),...
'interpreter','latex','fontsize',16);
%title('Gradient Descent with Fixed Learning Rate, a = 50','fontsize',15);
legend({sprintf('$\\alpha$ = %.5f',c),sprintf('$\\alpha$ = %.5f',c/2),sprintf('$\\alpha$ = 2/$(\\lambda_{min}+\\lambda_{max})$')},'fontsize',15,'Interpreter','latex')