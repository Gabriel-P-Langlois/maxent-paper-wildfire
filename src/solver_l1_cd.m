function [sol_w,sol_p,k] = solver_l1_cd(w,p0,lambda,A,Ed,max_iter,tol)
% Nonlinear PDHG method for solving Maxent with lambda*norm(\cdot)_{1}
% Input variables:
%   w: m x 1 vector -- Weights of the gibbs distribution.
%   p: n x 1 vector -- Parameterization of the gibbs probability
%   distribution, where p(j) = pprior(j)e^{u(j)-C}.
%   lambda: Positive number -- Hyperparameter.
%   A: n x m matrix -- Matrix of features (m) for each grid point (n).
%   tau, sigma, theta: Positive numbers -- Stepsize parameters.
%   Ed: m-dimensional vector -- Observed features of presence-only data. 
%   max_iter: Positive integer -- Maximum number of iterations.
%   tol:    Small number -- used for the convergence criterion

% Auxiliary variables I -- For the algorithm
tmp1 = ((p0'*A)') - Ed;
tmp2 = A*w;
wplus = w;

% Main algorithm
k = 0; flag_convergence = true(1);
d = zeros(length(w),1);

while (flag_convergence)
    % Update counter
    k = k + 1;
    
    % Apply thresholding
    for i=1:1:length(w)
       if(w(i)~=0)
           d(i) = lambda*sign(w(i)) + tmp1(i);
       elseif(abs(tmp1(i)) < lambda)
           d(i) = 0;
       else
           d(i) = -lambda*sign(w(i)) + tmp1(i);
       end
    end
    [~,j_ind] = max(abs(d));
    
    % Approximate steepest descent
    if(abs(w(j_ind)-tmp1(j_ind)) < lambda)
        eta = -w(j_ind);
    elseif(w(j_ind)-tmp1(j_ind) > lambda)
        eta = -(lambda+tmp1(j_ind));
    else
        eta = (lambda-tmp1(j_ind));
    end
    
    % Update dual variable + scalar product
    wplus(j_ind) = w(j_ind) + eta;
    tmp2 = tmp2 + eta*A(:,j_ind);

    % Update the probability + Different in expectations
    pplus = exp(tmp2-max(tmp2)); norm_sum = sum(pplus);
    tmp1 = ((pplus'*A)')/norm_sum - Ed; % Approximation technique? Can be used to approximate the other problem as well.
    
    w = wplus;

    % Convergence check -- We use the optimality condition on the l1 norm
    %disp(norm(tmp2,inf)/(lambda*(1+tol)))
    flag_convergence = ~(((k >= 40) && (norm(tmp1,inf) <= lambda*(1 + tol))) || (k >= max_iter));
end

% Final solutions
sol_p = pplus/norm_sum;
sol_w = wplus;
end