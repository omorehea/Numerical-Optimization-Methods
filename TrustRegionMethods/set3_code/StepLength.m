function a_ast = StepLength(p,x)

% Algorithm 3.5 in Numerical Optimization by Nocedal and Wright
% cost function and gradient are defined in cost_fun.m and grad_fun.m 

%%%%% input %%%%%%%
% p is the search direction (must be descent).
% x is the current point 

%%%%% output %%%%
% a_ast: the step size that satisfies the strong Wolfe conditions

%%%%% parameters %%%%
% a_max is an user provided parameter defines the maximum allowable step size.
% c1 and c2 are parameters in the Strong Wolfe Conditions
% 0<c1<c2<1
% For nonlinear conjugate gradient methods, set c2<0.5.

a_max = 1000;
c1 = 1e-4;
%c2 = 0.9; %for other methods
c2 = .9; %c2 < .5 for nonlinearCG    
a0 = 0;

%a_ast = 0.5*a_max; % initial guess
%a_ast = 1e+6; %for nonlinearCG       
a_ast = 1;
% for Newton's types of methods, set a_ast = 1.

phi0  = cost_fun(x);
dphi0 = grad_fun(x)'*p;
phi_pre = phi0;

i = 2;
while i < 100 % set the maximum number of iteration
    phi = cost_fun(x+a_ast*p);
    if phi > phi0 + c1*a_ast*dphi0 || (phi >= phi_pre && i > 2)
        a_ast = zoom(a0,a_ast,c1,c2,x,p,phi0,dphi0);
        break
    end
    dphi = grad_fun(x+a_ast*p)'*p;
    if abs(dphi) <= -c2*dphi0
        break
    end
    if dphi >= 0
        a_ast = zoom(a_ast,a0,c1,c2,x,p,phi0,dphi0);
        break
    end
    a0 = a_ast;
    a_ast = 0.5*(a_ast+a_max);
    phi_pre = phi;
    i = i + 1;
end

end

%%%%%%%%%%%%%%%%%%
function a = zoom(a_lo,a_hi,c1,c2,x,p,phi0,dphi0)
%Algorithm 3.6 in Numerical Optimization by Nocedal and Wright

phi_lo = cost_fun(x+a_lo*p);

counter = 1;
while counter < 100
    a = 0.5*(a_lo+a_hi); %set the try step using bisection
    phi = cost_fun(x+a*p);
    if phi > phi0 + c1*a*dphi0 || phi >= phi_lo
        a_hi = a;
    else
        dphi = grad_fun(x+a*p)'*p;
        if abs(dphi) <= -c2*dphi0
            break
        end
        if dphi*(a_hi-a_lo) >= 0
            a_hi = a_lo;
        end
        a_lo = a;
        phi_lo = phi;
    end
    counter = counter+1;
end

end % function zoom
