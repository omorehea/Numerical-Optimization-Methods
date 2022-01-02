%Trust Region algorithm based on standard Dogleg method

function p = subprob_standard_dogleg(g,B,delta)

temp = g'*B*g;  %||p^B||

p_b = -B\g;  %-inv(B)*g
p_u = -(g'*g)/temp*g;    %g'*g = ||g||^2 since g is a vector
    
    if norm(p_b) <= delta
        p = p_b;
    else
        if p_u >= delta
            p = -delta/norm(g)*g;
        else
            %find tau in [1,2] s.t ||p_u + (tau-1)*(p_b - p_u)||^2 = delta^2
            %solve with quadratic formula
            a = (p_b-p_u)'*(p_b-p_u); %= (||(p_b-p_u)||^2 since p vector
            b = 2*p_u'*(p_b-p_u);
            c = p_u'*p_u-delta^2;
            tau = 1 + (-b + sqrt(b^2 - 4*a*c))/(2*a);
            p = p_u + (tau - 1)*(p_b- p_u);
        end
    end
    

end


