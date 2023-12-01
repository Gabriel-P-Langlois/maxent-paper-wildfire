%% Description
% Nonlinear PDHG method for solving Maxent with elastic net penalty
    % Input:
    %   w_in: m x 1 vector -- Weights of the gibbs distribution.
    %   u_in: n x 1 vector -- Parameterization of the gibbs probability
    %   distribution, where p(j) = pprior(j)e^{u(j)-C}, C = normalizing
    %   constant
    %   t: Positive number -- Hyperparameter.
    %   A: n x m matrix -- Matrix of features (m) for each grid point (n).
    %   tau, sigma, theta: Positive numbers -- Stepsize parameters.
    %   Ed: m-dimensional vector -- Observed features of presence-only data. 
    %   max_iter: Positive integer -- Maximum number of iterations.
    %   tol:    Small number -- used for the convergence criterion
    
    % Output:
    %   w_out: m x 1 column vector -- dual solution
    %   p_out: n x 1 column vector -- primal solution
    %   num_iters: integer -- number of iterations

function [w_out,p_out,num_iters] = npdhg_solver_en_linear(w_in,u_in,...
    t,alpha,A,tau,sigma,theta,Ed,max_iters,tol)

    % Auxiliary variables, factors and variables
    num_iters = 0; 
    flag_convergence = true(1);
    m = length(w_in);

    factor1 = 1/(1+tau);
    factor2 = tau*factor1;
    wminus = w_in; wplus = w_in;
    u_in = u_in*factor1;
    
    % Main algorithm
    while (flag_convergence)
        % Update counter
        num_iters = num_iters + 1;
    
        % Update the primal variable
        % It's faster to loop over nonzero variables than use indices
        % when there are many zeros.
        for i=1:1:m
           if((w_in(i) ~= 0) || (wminus(i) ~= 0))
               tmp = factor2*(w_in(i) + theta*(w_in(i)-wminus(i)));
               u_in = u_in + A(:,i)*tmp;
           end
        end
        pplus = exp(u_in-max(u_in)); norm_sum = sum(pplus);
    
        % Update the dual variable -- Elastic net prox update
        tmp2 = (pplus.'*A).'; tmp2 = tmp2/norm_sum - Ed;
        wplus = proximal_operator_en(w_in-sigma*tmp2,sigma*t,alpha);
    
        % Convergence check -- Check that the optimality condition of the
        % elastic net penalty is satisfied after enough iterations
        %display(norm((1-alpha)*t*wplus + tmp2,inf) - alpha*t*(1+tol))
        flag_convergence = ~(((num_iters >= 40) && (norm((1-alpha)*t*wplus + tmp2,inf) <= ...
            max(tol,alpha*t*(1 + tol)) )) || (num_iters >= max_iters));
    
        % Increment variables
        u_in = u_in*factor1;
        wminus = w_in; w_in = wplus;
    end
    
    % Final solutions
    p_out = pplus/norm_sum;
    w_out = wplus;
end