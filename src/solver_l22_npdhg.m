function [sol_w,sol_z,sol_p,Ed_minus_Ep,k] = solver_l22_npdhg(w,z,lambda,A,tau,sigma,theta,Ed,max_iter,tol)
% Nonlinear PDHG method for solving Maxent with 0.5*normsq{\cdot}.
% Input variables:
%   w: m x 1 vector -- Weights of the gibbs distribution.
%   z: m x 1 vector -- Parameterization of the gibbs probability
%   distribution, where p(j) = pprior(j)e^{<z,Phi(j)>-C}.
%   lambda: Positive number -- Hyperparameter.
%   A: n x m matrix -- Matrix of features (m) for each grid point (n).
%   tau, sigma, gamma_h: Positive numbers -- Stepsize parameters.
%   Ed: m-dimensional vector -- Observed features of presence-only data. 
%   max_iter: Positive integer -- Maximum number of iterations.
%   tol:    Small number -- used for the convergence criterion

% Auxiliary variables
wminus = w;
factor1 = 1/(1+tau);
factor2 = 1/(1+lambda*sigma);

% Main algorithm
k = 0; flag_convergence = true(1);
while (flag_convergence)
    % Update counter
    k = k + 1;
    
    % Update the primal variable and the probability
    zplus = (z + tau*(w + theta*(w-wminus)))*factor1;
    
    % Compute pplus
    pplus = A*zplus;
    pplus = exp(pplus - max(pplus)); norm_sum = sum(pplus);
    
    % Update the dual variable
    temp2 = (pplus.'*A).';
    temp2 = Ed - temp2/norm_sum;
    wplus = factor2*(w + sigma*temp2);
 
    % Convergence check:
    flag_convergence = ~(((k >= 4) && (norm(temp2 - lambda*wplus,inf) < tol)) || (k >= max_iter));
    
    % Increment
    z = zplus;
    wminus = w; w = wplus;
    
%     % Value of the iterate -- comment out as needed
%     disp(['linf norm of the gradient of the primal problem:',num2str(norm(temp2 - lambda*wplus,inf)),'.'])
end

% Final solutions
sol_w = wplus;
sol_z = zplus;
sol_p = pplus/norm_sum;
Ed_minus_Ep = temp2;
end