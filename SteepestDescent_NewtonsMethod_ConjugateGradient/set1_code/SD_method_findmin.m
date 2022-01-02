%Steepest Descent Numerical Optimization Method
%Code to find the minimizer of the cost function
%Written by Owen Morehead, Oct 9 2021
%cost_fun() = cost function from cost_fun.m
%grad_fun() = gradient of cost function in grad_fun.m
%StepLength() = Step size that satisfy Wolfe conditions

clc;
clear all;

%global c  %constant in cost function. If no constants, comment out

max_iters = 1000; %maximum number of iterations
tol = 1e-15;      %tolerance level
dim = 2;          %dimensions of x
x_s = zeros(dim,max_iters);    %steepest descent solution at each iteration
x_errors = zeros(1,max_iters); %exact x value from cost function at each iteration

%for c = [1,10]
    x_initial = rand(2,1);  %random initial value of x 
    %cost_min = [2/c;-1];     %global minimizer of cost function found analytically
                            %written as 2x1 matrix = x*' (x_min transpose)
    x_s(:,1) = x_initial; %add the initial random value of x1 and x2 to our array of x values
    %x_errors(1) = norm(x_s(:,1)-cost_min); %add first element to x_errors, the error value between the point cost_min and x_initial
    p = -grad_fun(x_s(:,1)); %value of search direciton, p, evaluated at x_initial
    error_p = zeros(500,1);
    k = 1;
    while k < max_iters && norm(p) > tol  %Iterations will terminate when 
                                              %either k = max_iters, or 
                                              %p <= tolerance level
        a = StepLength(p,x_s(:,k));   %computing step length along p at each iteration
        x_s(:,k+1) = x_s(:,k) + a*p;  %line search iteration method
        %x_errors(k+1) = norm(x_s(:,k+1)-cost_min); %error in x between exact solution and numerical algorithm
        p = -grad_fun(x_s(:,k+1));    %value of p for next iteration step, k+1
        %error_p(k) = norm(p);
        k = k+1;

    end
    
    fprintf('Convergence at iteration k = %.2f\n',k)
    fprintf('Minimum found has value: (%.8f,%.8f) \n',x_s(1,end),x_s(2,end))
    hess_matrix_at_costmin = [2,-2;-2,9.391186];
    [V,D] = eig(hess_matrix_at_costmin);
    fprintf('Eigenvalue : %.5f ',diag(D));
    %figure(1);
    %grid on; hold on;
    %plot(x_s(1,:),x_s(2,:),'.');
    %plot(1:k,error_p) %This plot shows that the gradient converges to 0
    
%end
%xlabel('k'); ylabel('Log10(Error)');
%title(sprintf('SD Method with $x_0$ = (%.2f,%.2f)',x_initial(1),x_initial(2)),...
%'interpreter','latex','fontsize',16);
%legend('c = 1','c = 10');
%xlim([0,100]);

