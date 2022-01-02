%Trust Region algorithm based on Cauchy Point method

function p = subprob_cauchy(g,B,delta)

temp = g'*B*g;

if temp <= 0
    alpha_k = delta/norm(g);
else
    alpha_k = min(delta/norm(g),(norm(g)^2/temp));
end

p = -alpha_k*g;





end