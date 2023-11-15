%% Script written by Gabriel Provencher Langlois
% This script performs l1-regularized Maxent on the fire data set
% provided by JB using the nPDHG algorithm developed by GPL and JD.

% The nPDHG algorithm computes probabilities indirectly using the 
% parametrization
% p(j) = pprior(j)*exp(<z,Phi(j)>)/(sum_{j}pprior(j)*exp(<z,Phi(j)>))
% for some vector z. This parametrization is motivated from the optimality
% conditions of regularized-Maxent.

% Background and presence-only data are used for the uniform distribution.


%% Notes
% Convergence criterion is different for this problem than the one 
% for l22-regularized Maxent.

% Ergodic formulation.



%% Input
% Regularization path. Entries are a sequence of positive numbers starting
% from 1 and decreasing to some value > 0.
% This script solves l1-regularized Maxent with the hyperparameters
% reg_path*lambda in an efficient manner.

reg_path = 1:-0.1:0.50; 



%% Extract the data and prepare it
% block0_values: 38 Features (see bottom of the script)
% block1_values: 7 Features (see bottom of the script)
% block2_values: 5 quantities (see bottom of the script)

% The features and quantities are listed at the bottom of the script,
% after the code for the solver and before the misc information.

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
b = A(ind_fire_yes,:);

% Standardize the features.
% Note: Background features are normalized together.
%means_A = mean(A); A = A - means_A;
%sumsq_A = sum(A.^2); A = A./sqrt(sumsq_A/n); % Normalization
%b = A(ind_fire_yes,:);

% Average over the features of the presence only data. We take the average
% w.r.t. the uniform distribution with n1 elements.
% Note: We can weigh the background vs presence only data differently.
pempirical = single(zeros(n,1)); pempirical(ind_fire_yes) = 1/n1;
Ed = A'*pempirical;

% Compute the smallest parameter for which the dual solution is zero.
% Note: The prior distribution is uniform w.r.t. 
% to the background *AND* presence samples, i.e.,
% ones(n,1)/(n);
lambda_est = norm(Ed - A'*(ones(n,1)/n),inf);

% Compute hyperparameters to be used
lambda = lambda_est*reg_path;

% Clear irrelevant arrays
clear data8 ind_nan %ind_fire_yes



%% Parameters of the algorithm
% Dimensions
m = length(Ed);         % Number of features
l_max = length(lambda); % Length of the regularization path

% Placeholders for solutions
sol_npdhg_w = single(zeros(m,l_max));
sol_npdhg_u = single(zeros(n,l_max));
sol_npdhg_p = single(zeros(n,l_max)); sol_npdhg_p(:,1) = compute_p(sol_npdhg_u(:,1));

% Timings and Maximum number of iterations
time_npdhg_regular = 0;
time_npdhg_total = 0;
max_iter = 1000;



%% Script for the nPDHG algorithm
disp(' ')
disp('Algorithm: The nPGHG method (with regular sequence + variable selection)')

% Regularization path
time_L12 = 0;

% Store indices
% Call unregularized Maxent problem to estimate prob
% Perform variable selection with said probability...

for i=2:1:l_max
    tic
    t = lambda(i); 
    
    % Variable selection
    alpha = t/lambda(i-1);
    s = alpha*sol_npdhg_p(:,i-1) + (1-alpha)*pempirical;
    lhs = abs(Ed - A'*sol_npdhg_p(:,i-1));    
    rhs = lambda(i-1) - vecnorm(A,inf)'*sqrt(2*kldiv(s,sol_npdhg_p(:,i-1)))/alpha;
    ind = (lhs >= rhs);         % Indices that are non-zero.
    disp(['Percentage of coefficients found to be zero: ',num2str(100-100*sum(ind)/m)])
    
    % Initialize the regularization hyperparameter and other parameters
    tic
    L12_sq = max(sum((A(:,ind)').^2));
    time_L12 = time_L12 + toc;
    theta = 0;
    sigma = 0.5/L12_sq;
    tau = 2;

    % Call the solver for this problem and compute the resultant
    % probability distribution.    
    [sol_npdhg_w(ind,i),sol_npdhg_p(:,i),num_iter_tot_reg] = ... 
        npdhg_l1_solver(sol_npdhg_w(ind,i-1),log((n0+n1)*s),t,A(:,ind),tau,sigma,theta,Ed(ind),max_iter);   
    time_npdhg_regular = toc;
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_npdhg_regular),' seconds.'])
    disp(' ')
    time_npdhg_total = time_npdhg_total + time_npdhg_regular;
end
disp(['Total time elapsed for the nPDHG method = ',num2str(time_npdhg_total + time_L12),' seconds.'])
%% End of the script.



%% Auxiliary functions
function p = compute_p(u)
% Compute a probability vector p from the formula
% p(j) = exp([u]_{j})/(sum_{j}exp([u]_{j}))
% for every j in {1,...,n} and some parameter z.

a = u-max(u);
w = exp(a);
p = w/sum(w);
end

function kl_num = kldiv(prob1,prob2)
% Compute the Kullback--Leibler divergence between prob1 and prob2
    kl_num = prob1'*log(prob1./prob2);
end

%% Solver
function [sol_w,sol_p,num_iter_tot] = npdhg_l1_solver(w,u,lambda,A,tau,sigma,theta,Ed,max_iter)
% Nonlinear PDHG method for solving Maxent with 0.5*normsq{\cdot}.
% Input variables:
%   w = Array of dimension m x 1
%   u = Array of dimension n x 1
%   lambda = parameter > 0
%   A = An n x m matrix.
%   tau, sigma, gamma_h, m, d, max_iter = real numbers
%   Ed = Observed features of presence-only data. The features are averaged
%   w.r.t. a uniform distribution.

% Auxiliary variables
wminus = w;
p = exp(u-max(u));
p = p/sum(p);

factor1 = 1/(1+tau);
factor2 = tau*factor1;
u = u*factor1;

% Counter for the iterations
num_iter_tot = 0;

% Ergodic variables
sigma_0 = sigma;
Tminus = 0;
WKminus = zeros(length(w),1);
PKminus = zeros(length(p),1);

% Iterations
for k=1:1:max_iter
    num_iter_tot = num_iter_tot + 1;
    
    % Update the probability
    temp0 = factor2*(w + theta*(w-wminus));
    temp = A*temp0;
    
    uplus = u + temp;
    a = max(uplus);
    temp2 = exp(uplus-a);
    temp3 = sum(temp2); factor3 = 1/temp3;
    pplus = temp2*factor3;
    
    % Update the dual variable
    temp4 = A'*pplus;
    temp4 = temp4-Ed;
    wplus = w - sigma*(temp4);
    wplus = wthresh(wplus,'s',lambda*sigma);%sign(wplus).*max(0,abs(wplus) - lambda*sigma);
    
    % Ergodic variables
    T = Tminus + sigma/sigma_0;
    WK = (WKminus*Tminus + sigma*wplus/sigma_0)/T;
    PK = (PKminus*Tminus + sigma*pplus/sigma_0)/T;
    
    
    
    % Check for convergence (ERGODIC)
    disp(norm(temp4,inf)/(lambda + 1e-04))
    if((k >= 30) && (norm(temp4,inf) <= (lambda + 1e-04)))
        break
    end
    
    % Increment
    theta = 1/sqrt(1+tau); tau = theta*tau; sigma = sigma/theta;
    factor1 = 1/(1+tau);
    factor2 = tau*factor1;
    u = uplus*factor1;
    wminus = w; w = wplus;
    p = pplus;
    
    Tminus = T;
    WKminus = WK;
    PKminus = PK;
end

% Final solutions
sol_p = PK;
sol_w = WK;
%sol_w = wplus;
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