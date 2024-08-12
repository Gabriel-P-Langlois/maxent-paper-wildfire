function [w_out,p_out,num_iters] = npdhg_solver_en_linear(w_in,pprior,...
    t,alpha,A,tau,sigma,theta,Ed,max_iters,tol)

    % Auxiliary variables, factors and variables
    num_iters = 0; 
    flag_convergence = true(1);
    m = length(w_in);

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
    
        % Evaluate argument of the elastic net proximal operator
        tmp2 = (pplus.'*A).' - Ed;

        % Compute the EN proximal operator
        wplus = proximal_operator_en(w_in-sigma*tmp2,sigma*t,alpha);
    
        % Convergence check -- Check that the optimality condition of the
        % elastic net penalty is satisfied after enough iterations
        %display(norm((1-alpha)*t*wplus + tmp2,inf) - alpha*t*(1+tol))
        flag_convergence = ~convergence_criterion_en(num_iters,...
            max_iters,t,alpha,wplus,tmp2,tol);
    
        % Increment variables
        wminus = w_in; w_in = wplus;
    end
    
    % Final solutions
    p_out = pplus;
    w_out = wplus;
end