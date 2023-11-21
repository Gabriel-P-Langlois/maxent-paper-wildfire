%% Description
%
% This is the main function for solving the Elastic net regularized
% Maxent problem with the FISTA algorithm.


function [w_out,p_out,num_iters] = fista_solver_inf(w_in,pprior,lambda,...
    A,tau,mu,q,Ed,max_iter,tol)

    % Auxiliary variables, factors and variables
    num_iters = 0; 
    flag_convergence = true(1);

    % Initial quantities
    tk = 1;
    w_minus = w_in; wplus = w_in;

    while (flag_convergence)
        % Update counter
        num_iters = num_iters + 1;

        % Stepsies
        tkplus = 0.5*(1-q*(tk^2) + sqrt((1-q*(tk^2))^2 + 4*(tk^2)));
        beta = ((tk-1)/tkplus)*(1+mu*tau*(1-tkplus));

        % Compute next extragradient step
        yk = w_in + beta*(w_in-w_minus);

        % Evaluate probability from the extragradient step
        u = A*yk;
        pplus = pprior.*exp(u-max(u));
        pplus = pplus/sum(pplus);

        % Evaluate argument of the elastic net proximal operator
        tmp = (pplus.'*A).' - Ed;

        % Compute the EN proximal operator
        wplus = proximal_operator_inf(yk - tau*(tmp),lambda*tau);

        % Convergence check -- Check that the optimality condition of the
        % elastic net penalty is satisfied after enough iterations
        flag_convergence = ~(((num_iters >= 40) && (norm(tmp,1) <= ...
            lambda*(1 + tol))) || (num_iters >= max_iter));

        % Increment
        tk = tkplus;
        w_minus = w_in; w_in = wplus;

    end

    % Final solutions
    p_out = pplus;
    w_out = wplus;
end