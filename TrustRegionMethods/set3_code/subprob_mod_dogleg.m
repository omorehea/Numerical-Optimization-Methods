%Trust Region algorithm based on modified Dogleg method

function p = subprob_mod_dogleg(g,B,delta)

temp = g'*B*g;  %||p^B||
%p_s = -delta/norm(g)*g;  %minimum of linear model


if temp <= 0
    p = -delta/norm(g)*g;  %set p to be Cauchy point, this is the idea behind modified dogleg algorithm
else
    p_b = -B\g;  %-inv(B)*g
    p_u = -(g'*g)/temp*g;
    
    if p_u'*(p_b - p_u) <= 0 %first condition satisfied (g'*B*g > 0) but angle bewteen p_u and p_b is < 90 deg
        temp = norm(g)^2/temp;
        p = min(delta/norm(g),temp)*-g;
        %p = p_s;
    else
        %both condition satisfied, use standard dogleg method
        
    if norm(p_b) <= delta
        p = p_b;
    else
        if p_u >= delta
            p = -delta/norm(g)*g;
        else
            %find tau in [1,2] s.t ||p_u + (tau-1)*(p_b - p_u)||^2 = delta^2
            %solve with quadratic formula
            a = (p_b-p_u)'*(p_b-p_u); %= (||(p_b-p_u)||^2 since p vector
            b = 2*(p_b-p_u)'*p_u;
            c = p_u'*p_u-delta^2;
            tau = 1 + (-b + sqrt(b^2 - 4*a*c))/(2*a);
            p = p_u + (tau - 1)*(p_b- p_u);
        end
    end
    
    end
end
end


