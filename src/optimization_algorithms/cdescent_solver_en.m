%% Description
% Nonlinear PDHG method for solving Maxent with lambda*norm(\cdot)_{1}
% Input variables:
%   w_in: m x 1 vector -- Weights of the gibbs distribution.
%   p_in: n x 1 vector -- Parameterization of the gibbs probability
%   distribution, where p(j) = pprior(j)e^{u(j)-C}.
%   t: Positive number -- Hyperparameter.
%   alpha: parameter for the convexity between the l1 norm and l22 norms.
%   A: n x m matrix -- Matrix of features (m) for each grid point (n).
%   Ed: m-dimensional vector -- Observed features of presence-only data. 
%   max_iter: Positive integer -- Maximum number of iterations.
%   tol:    Small number -- used for the convergence criterion

% Output:
%   sol_w: m x 1 column vector -- dual solution
%   sol_p: n x 1 column vector -- primal solution

function [w_out,p_out,num_iters] = cdescent_solver_en(w_in,p_in,...
    t,alpha,A,Ed,max_iters,tol)

% Auxiliary variables -- For the algorithm
tmp1 = ((p_in'*A)') - Ed;
tmp2 = A*w_in;
wplus = w_in;

% Main algorithm
num_iters = 0; 
flag_convergence = true;
d = zeros(length(w_in),1);

% In the following, we need to compute estimate the stepsize of the 
% coordinate descent algorithm. This stepsize is derived from the analysis
% on pages 305-306 of Mohri et al. (2019) ML textbook.

while (flag_convergence)
    % Update counter
    num_iters = num_iters + 1;
    
    % Gradient of the smooth term of the object function
    tmp3 = tmp1 + (1-alpha)*t*w_in;

    % Compute descent directions
    for i=1:1:length(w_in)
       if(w_in(i)~=0)
           d(i) = (alpha*t)*sign(w_in(i)) + tmp3(i);
       elseif(abs(tmp3(i)) <= (alpha*t))
           d(i) = 0;
       else
           d(i) = -(alpha*t)*sign(w_in(i)) + tmp3(i);
       end
    end
    [~,j_ind] = max(abs(d));
    
    % Approximate steepest descent
    step_cond = w_in(j_ind) - tmp1(j_ind);

    if(abs(step_cond) <= (alpha*t))
        eta = -w_in(j_ind);
    elseif(step_cond > (alpha*t))
        eta = -((alpha*t)+tmp3(j_ind))/(1+t*(1-alpha));
    else
        eta = ((alpha*t)-tmp3(j_ind))/(1+t*(1-alpha));
    end
    
    % Update dual variable + scalar product
    wplus(j_ind) = w_in(j_ind) + eta;
    tmp2 = tmp2 + eta*A(:,j_ind);

    % Update the probability and the difference in expectations
    pplus = exp(tmp2-max(tmp2)); norm_sum = sum(pplus);
    tmp1 = ((pplus'*A)')/norm_sum - Ed;
    
    % Increment dual variable
    w_in = wplus;

    % Convergence check -- We use the optimality condition on the l1 normd
    flag_convergence = ~convergence_criterion_en(num_iters,...
            max_iters,t,alpha,wplus,tmp1,tol);
end

% Final solutions
p_out = pplus/norm_sum;
w_out = wplus;
end