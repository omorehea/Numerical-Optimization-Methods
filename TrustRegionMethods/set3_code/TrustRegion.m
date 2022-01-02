%Trust Region Methods, Numerical Optimization Method
%Written by Owen Morehead, Nov 19 2021
%cost_fun() = cost function from cost_fun.m
%grad_fun() = gradient of cost function in grad_fun.m
%StepLength() = Step size that satisfy Wolfe conditions

clc
clear all
close all

%--- Trust Region ---%

delta_max = 100;  %Max allowable delta
eta = 0.1;      %eta \in [0,1/4)

max_iters = 10000;  %maximum number of iterations
tol = 1e-12;      %tolerance level

global Q

n = 1000;    %dimension of decision variable
x = zeros(n,max_iters);  %store solutions for every iteration
delta = 1;  %initial size of trust region

error = zeros(1,max_iters);
cost = zeros(1,max_iters); 

Q = rand(n,n);
[Q,R] = qr(Q);    %QR factorization of Q
lambda = linspace(1,100,n);  %eigenvalues range from 1 to 100
Q = (Q'*diag(lambda)*Q)./n;   %generating symmetric and PD matrix Q with condition number 100
x(:,1) = 1*rand(n,1);

for k = 1:max_iters
    g = grad_fun(x(:,k));
    if norm(g,inf) < tol
        break
    end
    
    B = hessian_fun(x(:,k));
    
    %p = subprob_mod_dogleg(g,B,delta);
    p = subprob_cauchy(g,B,delta);
    %p = subprob_standard_dogleg(g,B,delta);
    
    %objective funciton, reduction of cost fun between iterations
    actual_reduction = cost_fun(x(:,k)) - cost_fun(x(:,k) + p); %f(x_k) - f(x_k + p_k)
    
    %model function, reduction between iterations
    predicted_reduction = -1*(g'*p + .5*p'*B*p);  %m_k(0) - m_k(p_k)
    
    if abs(predicted_reduction) < 1e-15 %to avoid unstability and NaN
        rho = 1;
    else
        rho = actual_reduction / predicted_reduction;
        
    end
    if rho < 0.25
        delta = 0.25*delta;  %delta_k+1 = .25*delta_k
        
    else
        if rho > 0.75 && abs(norm(p) - delta) < tol
            delta = min(2*delta,delta_max);
        end
    end

    if rho > eta
        x(:,k+1) = x(:,k) + p;
    else
        x(:,k+1) = x(:,k);
    end
    error(k) = norm(x(:,k),inf);
    cost(k) = cost_fun(x(:,k));
    delta;
end

figure(1);
grid on; hold on;
plot(1:k,log10(error(1:k)),'b*-');
title(sprintf('Trust Region Method'),'fontsize',15);
xlabel('k','Fontsize',15);
ylabel('log10(Error)','Fontsize',15);
legend({'Cauchy Point'},'fontsize',13)

