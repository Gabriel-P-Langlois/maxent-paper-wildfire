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