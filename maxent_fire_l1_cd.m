%% Script written by Gabriel Provencher Langlois
% This script performs l1-regularized Maxent on the fire data set
% provided by JB using the coordinate descent algorithm in the
% Cortes et al. (2015) paper: "Structural Maxent Models".



%% Notes
% Use optimality conditions of each subproblems? 
% In particular of the dual problem?

% Huh. It actually works!
% Significantly slower...

%% Input
% Regularization path stored as an array. The entries must be positive
% and decreasing numbers starting from one.
reg_path = [1,0.9,0.75,0.5,0.4,0.35,0.3:-0.025:0.175,0.1675,0.16:-0.005:0.125,0.1225:-0.0025:0.1];

% Tolerance for the optimality condition.
tol = 1e-04; 



%% Extract the data and prepare it
% Note: The features and quantities are listed at the bottom of the script,
% after the code for the solver and before the misc information.

% block0_values: 38 Features (see bottom of the script)
% block1_values: 7 Features (see bottom of the script)
% block2_values: 5 quantities (see bottom of the script)

% Read the data
A = [h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block0_values')',single(h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block1_values'))'];
data8 = h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block2_values')';

% Discard NaN values
ind_nan = isnan(A(:,1));
A = A(~ind_nan,:);
data8 = data8(~ind_nan,:);

% Separate data with fires and those with no fire.
ind_fire_yes = (data8(:,1) >= 1);

% Compute the number of background and presence-only data points
n0 = length(unique(data8(~ind_fire_yes,[2,4:5]),'rows'));
n1 = length(unique(data8(ind_fire_yes,[2,4:5]),'rows'));
n = n0 + n1;

% Scale the features
max_A = max(A);
min_A = min(A);
A = (A - min_A)./(max_A-min_A);

% Average over the features of the presence only data. We take the average
% w.r.t. the uniform distribution with n1 elements.
% Note: We can weigh the background vs presence only data differently.
pempirical = single(zeros(n,1)); pempirical(ind_fire_yes) = 1/n1;
Ed = A'*pempirical;

% Compute the smallest parameter for which the dual solution is zero.
% Note: The prior distribution is uniform w.r.t. to the background *AND* 
% presence samples. 
lambda_est = norm(Ed - A'*(ones(n,1)/n),inf);

% Compute hyperparameters to be used
lambda = lambda_est*reg_path;

% Clear irrelevant arrays
clear data8 ind_nan ind_fire_yes



%% Parameters of the algorithm
% Dimensions
m = length(Ed);         % Number of features
l_max = length(lambda); % Length of the regularization path

% Placeholders for solutions
sol_npdhg_w = single(zeros(m,l_max));
sol_npdhg_p = single(zeros(n,l_max)); sol_npdhg_p(:,1) = ones(n,1)/n;

% Timings and Maximum number of iterations
time_cd_regular = 0;
time_cd_total = 0;
max_iter = 5000;



%% Script for the coordinate descent algorithm
disp(' ')
disp('Algorithm: Coordinate Descent (Cortes et al. (2015)')

% Regularization path
for i=2:1:l_max
    tic
    t = lambda(i); 
    
    % Variable selection
    alpha = t/lambda(i-1);
    s = alpha*sol_npdhg_p(:,i-1) + (1-alpha)*pempirical;
    lhs = abs(Ed - A'*sol_npdhg_p(:,i-1));    
    rhs = lambda(i-1) - vecnorm(A,inf)'*sqrt(2*kldiv(s,sol_npdhg_p(:,i-1)))/alpha;
    ind = (lhs >= rhs);         % Indices that are non-zero.
    
    % Display coefficients found to be zero.
    disp(['Percentage of coefficients found to be zero: ',num2str(100-100*sum(ind)/m)])

    % Call the solver for this problem and compute the resultant
    % probability distribution.    
    [sol_npdhg_w(ind,i),sol_npdhg_p(:,i),num_iter_tot_reg] = ... 
        cd_l1_solver(sol_npdhg_w(ind,i-1),s,t,A(:,ind),Ed(ind),max_iter,tol);   
    time_cd_regular = toc;
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_cd_regular),' seconds.'])
    disp(' ')
    time_cd_total = time_cd_total + time_cd_regular;
end
disp(['Total time elapsed for the CD method = ',num2str(time_cd_total),' seconds.'])
% End of the script.



%% Auxiliary functions
% Kullback--Leibler divergence
function kl_num = kldiv(prob1,prob2)
    kl_num = prob1'*log(prob1./prob2);
end



%% Solver
function [sol_w,sol_p,k] = cd_l1_solver(w,p0,lambda,A,Ed,max_iter,tol)
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