function [sol_w,sol_p,k] = solver_l1_npdhg(w,u,lambda,A,tau,sigma,theta,Ed,max_iter,tol)
% Nonlinear PDHG method for solving Maxent with lambda*norm(\cdot)_{1}
% Input:
%   w: m x 1 vector -- Weights of the gibbs distribution.
%   u: n x 1 vector -- Parameterization of the gibbs probability
%   distribution, where p(j) = pprior(j)e^{u(j)-C}.
%   lambda: Positive number -- Hyperparameter.
%   A: n x m matrix -- Matrix of features (m) for each grid point (n).
%   tau, sigma, theta: Positive numbers -- Stepsize parameters.
%   Ed: m-dimensional vector -- Observed features of presence-only data. 
%   max_iter: Positive integer -- Maximum number of iterations.
%   tol:    Small number -- used for the convergence criterion

% Output:
%   sol_w: m x 1 column vector -- dual solution
%   sol_p: n x 1 column vector -- primal solution

% Multiplicative factors
factor1 = 1/(1+tau);
factor2 = tau*factor1;

% Auxiliary variables I -- For the algorithm
wminus = w; wplus = w;
%p0 = exp(u-max(u)); sum_p0 = sum(p0);
%tmp2 = ((p0'*A)')/sum_p0 - Ed;


% Main algorithm
k = 0; flag_convergence = true(1);
m = length(w);
u = u*factor1;
while (flag_convergence)
    % Update counter
    k = k + 1;
                          
    % Update the primal variable
    for i=1:1:m
       if((w(i) ~= 0) || (wminus(i) ~= 0))
           tmp = factor2*(w(i) + theta*(w(i)-wminus(i)));
           u = u + A(:,i)*tmp;
       end
    end
    pplus = exp(u-max(u)); norm_sum = sum(pplus);

    % Update the dual variable I -- Compute new expected value
    tmp2 = (pplus.'*A).';
    tmp2 = tmp2/norm_sum - Ed;
    
    % Update the dual variable II -- Thresholding
    wplus = w - sigma*tmp2;
    temp3 = (abs(wplus)-sigma*lambda); temp3 = 0.5*(temp3+abs(temp3));
    wplus = sign(wplus).*temp3;
    
    % Convergence check -- We use the optimality condition on the l1 norm
    flag_convergence = ~(((k >= 40) && (norm(tmp2,inf) <= lambda*(1 + tol))) || (k >= max_iter));
    
    % Increment parameters, factors, nonzero indices, and variables.
    theta = 1/sqrt(1+tau); tau = theta*tau; sigma = sigma/theta;
    
    % Multiplicative factors
    factor1 = 1/(1+tau);
    factor2 = tau*factor1;
    
    % Variables
    u = u*factor1;
    wminus = w; w = wplus;
end

% Final solutions
sol_p = pplus/norm_sum;
sol_w = wplus;
end