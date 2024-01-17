%% Description
% Nonlinear PDHG method for solving Maxent with lambda*norm(\cdot)_{1}
    % Input:
    %   w_in: m x 1 vector -- Weights of the gibbs distribution
    %   pprior: n x 1 vector -- Prior distribution
    %   t: Positive number -- Hyperparameter
    %   groups: Indices of the groups for this solver
    %   A: n x m matrix -- Matrix of features (m) for each grid point (n)
    %   tau, sigma, theta: Positive numbers -- Stepsize parameters.
    %   Ed: m-dimensional vector -- Observed features of presence-only data 
    %   max_iter: Positive integer -- Maximum number of iterations.
    %   tol:    Small number -- used for the convergence criterion
    
    % Output:
    %   w_out: m x 1 column vector -- dual solution
    %   p_out: n x 1 column vector -- primal solution
    %   num_iters: integer -- number of iterations

    function [w_out,p_out,num_iters] = npdhg_solver_nogl(w_in,pprior,...
    t,groups,A,tau,sigma,theta,Ed,max_iters,tol)

    % Auxiliary variables, factors and variables
    num_iters = 0; 
    flag_convergence = true(1);
    wminus = w_in; wplus = w_in; tmp = w_in;
    
    % Main algorithm
    while (flag_convergence)
        % Update counter
        num_iters = num_iters + 1;
    
        % Update the primal variable
        tmp = (tmp + tau*(w_in + theta*(w_in-wminus)))/(1+tau);
        u = A*tmp;

        pplus = pprior.*exp(u-max(u)); 
        pplus = pplus/sum(pplus);
    
        % Update the dual variable -- NOGL update
        tmp2 = (pplus.'*A).' - Ed;
        wplus = proximal_operator_nogl(w_in-sigma*tmp2,sigma*t,groups);
    
        % Convergence check -- Check that the optimality condition of the
        % elastic net penalty is satisfied after enough iterations
        flag_convergence = ~convergence_criterion_nogl(num_iters,max_iters,...
                t,groups,tmp2,tol);

        % Increment parameters
        theta = 1/sqrt(1+tau); tau = theta*tau; sigma = sigma/theta;
    
        % Increment variables
        wminus = w_in; w_in = wplus;
    end
    
    % Final solutions
    p_out = pplus;
    w_out = wplus;
end