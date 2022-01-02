%Limited Memory BFGS, Numerical Optimization Method
%Written by Owen Morehead, Nov 5 2021
%cost_fun() = cost function from cost_fun.m
%grad_fun() = gradient of cost function in grad_fun.m
%StepLength() = Step size that satisfy Wolfe conditions

clc
clear all
close all

%--- Limited Memory BFGS ---%

max_iters = 1000;  %maximum number of iterations
tol = 1e-10;      %tolerance level
dim = 1;          %dimensions of x

global ALPHA
ALPHA = 100;
n = 1000;

cost_min = ones(n,1);    %global minimizer of cost function
                         %written as nx1 matrix = x*' (x_min transpose)
                
x_initial = (-1)*ones(n,1);  %initial value of x

m = max_iters;   %m variable for limited memory algorithm
S = zeros(n,m); %initializing S variable
Y = zeros(n,m); %initializing Y variable
error = zeros(n,1);

x = x_initial;  %only need x for iteration k at each iteration so can set dim x = [n,1]
grad = grad_fun(x);
error(1) = norm(x - cost_min,inf);
p = -grad; %initial search direction is steepest descent direction
runtime = zeros(n,1);
k = 1;
while k < max_iters && error(k)>tol
    tStart = tic;
    alpha = StepLength(p,x);
    s = alpha*p;  %s_k
    x = x + s;    %x_k+1
    
    grad_new = grad_fun(x);
    y = grad_new - grad;
    grad = grad_new;
    
    
    if k == 1
        gamma = (y'*s)/(y'*y);
        if gamma < 0
            gamma = 1;
        end
    end
    
    if k <= m
        memory = k;
        S(:,k) = s;
        Y(:,k) = y;
        
        
    else
        
        memory = m;
        gamma = (y'*s)/(y'*y);
        S = [S(:,2:m),s];
        Y = [Y(:,2:m),y];
    end
    
    p = -1*LBFGSrec(S(:,1:memory),Y(:,1:memory),gamma,grad,memory);
    %p'*grad_new; value of the inner product
    k = k + 1;
    error(k) = norm(x-cost_min,inf);
    runtime(k) = runtime(k-1) + toc(tStart);
    
end

figure(1);
grid on; hold on;
plot(1:k,log10(error(1:k)),'b*-');

figure(2);
subplot(2,1,1); grid on; hold on;
plot(runtime(1:k),log10(error(1:k)),'b*-');
title(sprintf('Run Time Comparison: m = %.1f',m),'fontsize',15);
ylabel('log10(Error)','Fontsize',15);
legend({'Limited Memory BFGS'},'fontsize',13,'Location','southwest')

%--- Standard BFGS ---%
%H_0 = eye(n)

x = x_initial;
k = 1;
H = eye(n);
grad = grad_fun(x);
error = zeros(n,1);
error(1) = norm(x - cost_min,inf);
runtime = zeros(n,1);

while error(k) > tol && k < max_iters
    tStart = tic;
    k;
    p = -H*grad;
    p;
    alpha = StepLength(p,x);
    s = alpha*p;
    x = x + alpha*p;
    
    grad_new = grad_fun(x);
    y = grad_new - grad;
    grad = grad_new;
    
    if k == 1
        beta = (y'*s)/(y'*y);
        if beta > 0
            H = beta*eye(n);
        end
    end
    rho = 1/(s'*y);
    t1 = rho*s;
    t2 = H - (H*y)*t1';
    %H = t2 - t1*(y'*t2) + t1*s';
    H = (eye(n) - rho*s*y')*H*(eye(n) - rho*y*s') + rho*s*(s');
    k = k+1;
    error(k) = norm(x - cost_min,inf);
    runtime(k) = runtime(k-1) + toc(tStart);
end

figure(1);
plot(1:k,log10(error(1:k)),'r.-');
xlabel('k'); ylabel('Log10(Error)');
title('BFGS Methods on Extended Rosenbrock Function','fontsize',15);
legend({'Limited Memory BFGS','Standard BFGS'},'fontsize',15)

figure(2);
subplot(2,1,2);
plot(runtime(1:k),log10(error(1:k)),'r*-');
xlabel('runtime','Fontsize',15); ylabel('log10(Error)','Fontsize',15);
legend({'Standard BFGS'},'fontsize',13,'Location','southwest')
