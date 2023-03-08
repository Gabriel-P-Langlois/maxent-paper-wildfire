%% Script written by Gabriel Provencher Langlois
% This script performs l1-regularized Maxent on the fire data set
% provided by JB using the nPDHG algorithm developed by GPL and JD.

% The nPDHG algorithm computes probabilities indirectly using the 
% parametrization
% p(j) = pprior(j)*exp(u(j) - c)
% for some vector u. This parametrization is motivated from the optimality
% conditions of regularized-Maxent.



%% Notes
% Speedups.
%   - The dual update of the nPDHG algorithm gives a vector that
%     frequently has many entries equal to zero. This script uses this fact
%     to avoid unnecessary computations of the form A(:,i)*w(i) when
%     w(i) = 0. This is important when the matrix A is large because the
%     speedup can be enormous.
%   - The script employs a loop to perform this for the primal update. 
%     This is not ideal, but it does not matter that much for this problem 
%     because the vector of features is small (45 for this problem). 
%     Subindexing in MATLAB is too slow in comparison. 
%     A C++ code for speeding this up would be ideal.
%   - A similar trick for the dual update does not work very well.


% Main bottlenecks are the the computation of pplus and the dual variable.

% It took ~ 3199.7547 seconds for GPL's laptop to process 
% the regularization path
% [1,0.9,0.75,0.5,0.4,0.35,0.3:-0.025:0.175,0.1675,0.16:-0.005:0.125,0.1225:-0.0025:0.1].

%% Input
% Regularization path stored as an array. The entries must be positive
% and decreasing numbers starting from one.
reg_path = [1,0.998,0.996,0.994]; %[1,0.9,0.75,0.5,0.4,0.35,0.3:-0.025:0.175,0.1675,0.16:-0.005:0.125,0.1225:-0.0025:0.1];

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

% Obtain regional indices
reg_index = data8(:,3);
ind_r_zero = (reg_index == 0);

% Discard data whose regional index is 0.
reg_index(reg_index == 0) = [];
A = A(~ind_r_zero,:);
data8 = data8(~ind_r_zero,:);

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

% Compute priors for the different regions + prior for the algorithm
r = double(max(reg_index));
nr = zeros(r,1);
prior_r = zeros(r,1);
pprior = zeros(n,1);
for i=1:r
    nr(i) = double(sum(reg_index == i));
    prior_r(i) = (1/(nr(i)*r));
    pprior(reg_index == i) = prior_r(i);
end

% Define the empirical distribution II: Allow absence points to be nonzero
% Use something similar to the prior
pempirical_r = zeros(r,1); pempirical_r2 = zeros(r,1);
pempirical = zeros(n,1); sum_pempirical = 0;
for i=1:r
    n1_r = length(unique(data8(and(ind_fire_yes, (reg_index == i)),[2,4:5]),'rows'));
    pempirical_r(i) = (1/(nr(i)));
    pempirical_r2(i) = (1/(r*nr(i)));
    sum_pempirical = sum_pempirical + pempirical_r(i)*n1_r + pempirical_r2(i)*double(sum(and(~ind_fire_yes,reg_index == i)));
end
pempirical_r = pempirical_r/sum_pempirical;
pempirical_r2 = pempirical_r2/sum_pempirical;

for i=1:r
    pempirical(and(reg_index == i,ind_fire_yes),:) = pempirical_r(i);
    pempirical(and(reg_index == i,~ind_fire_yes),:) = pempirical_r2(i);
end

% Define its expected value
Ed = A'*pempirical;

% Compute the smallest parameter for which the dual solution is zero.
lambda_est = norm(Ed - A'*pprior,inf);

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
sol_npdhg_p = single(zeros(n,l_max)); sol_npdhg_p(:,1) = pprior;

% Timings and Maximum number of iterations
time_npdhg_regular = 0;
time_npdhg_total = 0;
max_iter = 2000;

% Other quantities
theta = 0;
tau = 2;

%% Script for the nPDHG algorithm
disp(' ')
disp('Algorithm: The nPGHG method (with regular sequence + variable selection)')

% Regularization path
time_L12 = 0;
for i=2:1:l_max
    tic
    t = lambda(i); 
    
    % Variable selection
    alpha = t/lambda(i-1);
    s = alpha*sol_npdhg_p(:,i-1) + (1-alpha)*pempirical;
    lhs = abs(Ed - A'*sol_npdhg_p(:,i-1));    
    disp(kldiv(s,sol_npdhg_p(:,i-1)))
    rhs = lambda(i-1) - vecnorm(A,inf)'*sqrt(2*kldiv(s,sol_npdhg_p(:,i-1)))/alpha;
    ind = (lhs >= rhs);         % Indices that are non-zero.
    
    % Display coefficients found to be zero.
    disp(['Percentage of coefficients found to be zero: ',num2str(100-100*sum(ind)/m)])
    
    % Initialize the regularization hyperparameter and other parameters
    tic
    L12_sq = max(sum((A(:,ind)').^2));
    time_L12 = time_L12 + toc;
    sigma = 0.5/L12_sq;

    % Call the solver for this problem and compute the resultant
    % probability distribution.    
    [sol_npdhg_w(ind,i),sol_npdhg_p(:,i),num_iter_tot_reg] = ... 
        npdhg_l1_solver(sol_npdhg_w(ind,i-1),log((n0+n1)*s),t,A(:,ind),tau,sigma,theta,Ed(ind),max_iter,tol);   
    time_npdhg_regular = toc;
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_npdhg_regular),' seconds.'])
    disp(' ')
    time_npdhg_total = time_npdhg_total + time_npdhg_regular;
end
disp(['Total time elapsed for the nPDHG method = ',num2str(time_npdhg_total + time_L12),' seconds.'])
% End of the script.



%% Auxiliary functions
% Kullback--Leibler divergence
function kl_num = kldiv(prob1,prob2)
    kl_num = prob1'*log(prob1./prob2);
end



%% Solver
function [sol_w,sol_p,k] = npdhg_l1_solver(w,u,lambda,A,tau,sigma,theta,Ed,max_iter,tol)
% Nonlinear PDHG method for solving Maxent with lambda*norm(\cdot)_{1}
% Input variables:
%   w: m x 1 vector -- Weights of the gibbs distribution.
%   u: n x 1 vector -- Parameterization of the gibbs probability
%   distribution, where p(j) = pprior(j)e^{u(j)-C}.
%   lambda: Positive number -- Hyperparameter.
%   A: n x m matrix -- Matrix of features (m) for each grid point (n).
%   tau, sigma, theta: Positive numbers -- Stepsize parameters.
%   Ed: m-dimensional vector -- Observed features of presence-only data. 
%   max_iter: Positive integer -- Maximum number of iterations.
%   tol:    Small number -- used for the convergence criterion

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
    tmp2 = (pplus'*A)';
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


%% Information regarding the wildfire data set.
% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block0_items')
% 
% ans =
% 
%   38×1 cell array
%     {'Tmax       '}
%     {'VPD        '}
%     {'Prec       '}
%     {'Solar      '}
%     {'Wind       '}
%     {'Elev       '}
%     {'RH         '}
%     {'FM1000     '}
%     {'Ant_Tmax   '}
%     {'AvgVPD_3mo '}
%     {'Avgprec_3mo'}
%     {'Ant_RH     '}
%     {'CAPE       '}
%     {'FFWI_max3  '}
%     {'FFWI_max7  '}
%     {'Tmin       '}
%     {'Camp_dist  '}
%     {'Camp_num   '}
%     {'Road_dist  '}
%     {'Avgprec_4mo'}
%     {'Avgprec_2mo'}
%     {'VPD_max3   '}
%     {'VPD_max7   '}
%     {'Tmax_max3  '}
%     {'Tmax_max7  '}
%     {'Tmin_max3  '}
%     {'Tmin_max7  '}
%     {'Slope      '}
%     {'Southness  '}
%     {'AvgVPD_4mo '}
%     {'AvgVPD_2mo '}
%     {'SWE_mean   '}
%     {'SWE_max    '}
%     {'AvgSWE_3mo '}
%     {'Delta_T    '}
%     {'Biomass    '}
%     {'Lightning  '}
%     {'RH_min3    '}

% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block1_items')
% 
% ans =
% 
%   7×1 cell array
% 
%     {'Antprec_lag1'}
%     {'Forest      '}
%     {'Grassland   '}
%     {'Urban       '}
%     {'Antprec_lag2'}
%     {'Popdensity  '}
%     {'Housedensity'}

% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block2_items')
% 
% ans =
% 
%   5×1 cell array
% 
%     {'fire_freq'}
%     {'month    '}
%     {'reg_indx '}
%     {'X        '}
%     {'Y        '}

%% Unused code

%%%% Original primal update
%     % Update the primal variable
%      temp(ind_primal) = factor2*(w(ind_primal) + theta*(w(ind_primal)-wminus(ind_primal)));
%      u = u + A(:,ind_primal)*temp(ind_primal);

%%%% Original dual update
%   tmp2 = (pplus'*A)';
%   tmp2 = tmp2 - Ed;

%%%% Obsolete stuff
    % Update the dual variable I -- Loop + allocation
%     if(mod(k,10) == 0) % Every 10 iterations, perform a full update
%         p0 = pplus;
%         tmp2 = ((p0'*A)')/norm_sum - Ed;
%         test_fac = tmp2;
%     else % Determine nonzero entries 
%         idx_dual = all_ind((sigma*lambda - sigma*sum(abs(pplus/norm_sum-p0)) - abs(w - sigma*test_fac) < 0));
%         tmp2 = zeros(length(w),1);
%         tmp2(idx_dual) = -Ed(idx_dual);
%         for i=idx_dual
%             tmp2(i) = tmp2(i) + pplus'*A(:,i);
%         end
%     end


    %temp0 = Ed - (p0'*A)';
%     uplus = u;
%     idx_primal = ind_primal.*(1:m); idx_primal(idx_primal == 0) = [];
%     for l=idx_primal
%         temp = factor2*(w(l) + theta*(w(l)-wminus(l)));
%         uplus = uplus +  (A(:,l).'*temp)';
%     end

    
    % NOTE: Potential gain in speed if we determine in advance which
    % features we need to compute.
    % The factor (pplus'*A(:,ind_dual))'; is ineffective...
    
%     ind_dual = (sigma*lambda - abs(w + sigma*temp0) - sigma*sum(abs(pplus-p0)) < 0)';
%     ind_dual = ind_dual.*(1:m); ind_dual(ind_dual == 0) = [];
%     
%     % Update the dual variable
%     wplus = zeros(length(w),1);
%     
%     for j=ind_dual
%         wplus(l) = w(l) - sigma*(pplus'*A(:,l) - Ed(l));
%     end
% 
% p0 = exp(u-max(u));
% p0 = p0/sum(p0);
% m = length(w);