%% Description
% Nonlinear PDHG method for solving Maxent with lambda*norm(\cdot)_{1}
    % Input:
    %   w_in: m x 1 vector -- Weights of the gibbs distribution.
    %   u_in: n x 1 vector -- Parameterization of the gibbs probability
    %   distribution, where p(j) = e^{u(j)-C}, C = normalizing
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

function [w_out,p_out,num_iters] = npdhg_solver_en_sublinear(w_in,u_in,...
    t,alpha,A,tau,sigma,theta,Ed,max_iters,tol)

    % Auxiliary variables, factors and variables
    num_iters = 0; 
    flag_convergence = true(1);

    factor1 = 1/(1+tau);
    factor2 = tau*factor1;
    wminus = w_in; wplus = w_in;
    u_in = u_in*factor1;
    
    % Main algorithm
    while (flag_convergence)
        % Update counter
        num_iters = num_iters + 1;
    
        % Update the primal variable
        tmp = factor2*(w_in + theta*(w_in-wminus));
        u_in = u_in + A*tmp;

        pplus = exp(u_in-max(u_in)); 
        pplus = pplus/sum(pplus);
    
        % Evaluate argument of the elastic net proximal operator
        tmp2 = (pplus.'*A).' - Ed;

        % Compute the EN proximal operator
        wplus = proximal_operator_en(w_in-sigma*tmp2,sigma*t,alpha);
    
        % Convergence check -- Check that the optimality condition of the
        % elastic net penalty is satisfied after enough iterations
        flag_convergence = ~convergence_criterion_en(num_iters,...
            max_iters,t,alpha,wplus,tmp2,tol);
    


        % Increment parameters
        theta = 1/sqrt(1+tau); tau = theta*tau; sigma = sigma/theta;
    
        % Increment multiplicative factors
        factor1 = 1/(1+tau);
        factor2 = tau*factor1;
    
        % Increment variables
        u_in = u_in*factor1;
        wminus = w_in; w_in = wplus;
    end
    
    % Final solutions
    p_out = pplus;
    w_out = wplus;
end