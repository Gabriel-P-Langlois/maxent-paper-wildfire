%% Script written by Gabriel Provencher Langlois
% This script performs l1-regularized Maxent on the fire data set
% provided by JB using the nPDHG algorithm developed by GPL and JD.

% The nPDHG algorithm computes probabilities indirectly using the 
% parametrization
% p(j) = pprior(j)*exp(u(j) - c)
% for some vector u. This parametrization is motivated from the optimality
% conditions of regularized-Maxent.



%% Notes
% Speedups:
%   - Use direct indexing, if possible. If not, use logical indexing.
%   - Subindexing can be costly. Loops might be better for this!
%     Either that, or write C-code to speed up the computation of the
%     primal variable. (Loop over columns if possible.)



%% Input
% Regularization path stored as an array. The entries must be positive
% and decreasing numbers starting from one.
reg_path = [1,0.9,0.75,0.5,0.4,0.35,0.3:-0.025:0.175,0.1675,0.16:-0.005:0.125,0.1225:-0.0025:0.1];

% Note: the regularization path
%[1,0.9,0.75,0.5,0.4,0.35,0.3:-0.025:0.175,0.1675,0.16:-0.005:0.125,0.1225:-0.0025:0.1] 
% took 3199.7547 seconds to run (31 values). 

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
function [sol_w,sol_p,num_iter_tot] = npdhg_l1_solver(w,u,lambda,A,tau,sigma,theta,Ed,max_iter,tol)
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

% Counter for the iterations
num_iter_tot = 0;

% Multiplicative factors
factor1 = 1/(1+tau);
factor2 = tau*factor1;

% Auxiliary variables I
wminus = w; wplus = w;
u = u*factor1;  % For the algorithm

% Auxiliary variable II -- For indexing nonzero quantities
ind_w = (w~=0); indminus_w = ind_w;

% Main algorithm
for k=1:1:max_iter
    % Update counter
    num_iter_tot = num_iter_tot + 1;
    
    % Determine nonzero indices for the primal variable update.
    ind_primal = or(ind_w,indminus_w);
    ind_primal = ind_primal'.*(1:1:length(ind_primal));
    ind_primal(ind_primal == 0) = [];
                          
    % Loop over nonzero indices.
    % Note (GPL): Yes, loops are inefficient in matlab, but this is one of 
    % the notable exception. Reading through the nonzero indices all at 
    % once is much slower than looping over the nonzero indices. 
    % This probably can be improved using some C++ code.
    for i=ind_primal
        temp = factor2*(w(i) + theta*(w(i)-wminus(i)));
        u = u + A(:,i)*temp;
    end
    
    % Update the primal variable
    pplus = exp(u-max(u));
    pplus = pplus/sum(pplus);

    % Update the dual variable 
    % Note: Potential speed up here if we know in advance which
    % coefficients are zero.
    temp2 = (pplus'*A)';
    temp2 = temp2-Ed;
    wplus = w - sigma*temp2;
    
    % Soft thresholding
    temp3 = (abs(wplus)-sigma*lambda); temp3 = 0.5*(temp3+abs(temp3));
    wplus = sign(wplus).*temp3;
    
    % Convergence check -- We use the optimality condition on the l1 norm
    if((k >= 40) && (norm(temp2(ind_primal),inf) <= lambda*(1 + tol)))
        break
    end
    
    % Increment parameters, factors, nonzero indices, and variables.
    theta = 1/sqrt(1+tau); tau = theta*tau; sigma = sigma/theta;
    
    % Multiplicative factors
    factor1 = 1/(1+tau);
    factor2 = tau*factor1;
    
    % Identify nonzero indices
    indminus_w = ind_w;
    ind_w = wplus~=0;
    
    % Variables
    u = u*factor1;
    wminus = w; w = wplus;
end

% Final solutions
sol_p = pplus;
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

%%%% Obsolete stuff
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