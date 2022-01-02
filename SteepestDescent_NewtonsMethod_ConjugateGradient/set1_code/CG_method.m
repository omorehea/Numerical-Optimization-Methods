%Conjugate Gradient Method, Numerical Optimization
%Written by Owen Morehead, Oct 22 2021


clc
clear all
close all

max_iters = 1000; %maximum number of iterations
tol = 1e-16;      %tolerance level
n = 1000;         %dimension of matrix A
m = 3;           

%lambda = linspace(10,1000,n); %uniformly distributed eigenvalues from 10 to 1000
%lambda1 = linspace(9,11,m); %first m eigenvalues distributed between 9 and 11
%lambda2 = linspace(999,1001,max_iters-m); %rest of n-m eigenvalues distributed between 999 and 1001
epsilon = 1e-4;
lambda1 = 10 + epsilon*(rand(1,m));
lambda2 = 1000 + epsilon*(rand(1,n-m));

lambda = sort([lambda1,lambda2]);

[Q,R] = qr(rand(n,n)); %QR factorization to create a random orthogonal matrix Q
A = Q'*diag(lambda)*Q; %matrix a is symmetric, positive def

x_g = rand(n,1); %random guess for optimal solution
b = A*x_g;       %b = Ax^*
x = zeros(n,1);
error = zeros(max_iters,1);
error(1) = sqrt((x-x_g)'*A*(x-x_g)); 

k = 0;  
r = A*x - b;  %residual 
p = -r;       %search direction p_k
r_norm = r'*r;   %to be used in algorithm below

analytic_error = zeros(max_iters,1);
analytic_error(1) = sqrt((x-x_g)'*A*(x-x_g));

while norm(r,inf) > tol && k < max_iters
    alpha = r_norm/(p'*A*p);   %calculate step size
    x = x + alpha*p;           %equation for x_{k+1} = x_k + aplha_k*p_k
    r = r + alpha*A*p;         %r_{k+1} = r_{k} + ...
    r_norm_next = r'*r;        
    beta = r_norm_next/r_norm; %beta_{k+1} = r'_{k+1}*r_{k+1}/(r'_k*r_k)
    p = beta*p -r;             %p_{k+1} = beta_{k+1}*p_k - r_{k+1}
    r_norm = r_norm_next;      %redefining r_norm for next iteration
    k = k+1;
    error(k) = sqrt((x-x_g)'*A*(x-x_g));  %error at each iteration
                                          %error = ||x-x^*||^2_A
    %analytic_error(k) = 2 * ((sqrt(max(lambda)/min(lambda)) - 1)/...
        %(sqrt(max(lambda)/min(lambda)) + 1))^k * error(1);                                 
end

grid on; hold on;
plot(1:k,log10(error(1:k)),'.-'); %numerical result
%plot(1:k,log10(analytic_error(1:k)),'.'); %analytical comparision
xlabel('k'); ylabel('Log10(Error)');
title('Conjugate Gradient Method with Clustered Eigenvalues, \epsilon = 1e-4','fontsize',15);
%legend({'Numerical Error','Analytical Error'},'fontsize',15)