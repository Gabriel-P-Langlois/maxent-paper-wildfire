%% Script written by Gabriel Provencher Langlois
% This script performs l1-regularized Maxent on the fire data set
% provided by JB using the nPDHG algorithm developed by GPL and JD.

% The nPDHG algorithm computes probabilities indirectly using the 
% parametrization
% p(j) = pprior(j)*exp(<z,Phi(j)>)/(sum_{j}pprior(j)*exp(<z,Phi(j)>))
% for some vector z. This parametrization is motivated from the optimality
% conditions of regularized-Maxent.


%% Notes
% Convergence criterion is different for this problem than the l22.

% We can only apply a variable selection technique if we use both
% background and presence data for the approximation of the spatial
% density (?).

%% Input
% Regularization path. Entries are a sequence of positive numbers starting
% from 1 and decreasing to some value > 0.
% This script solves l1-regularized Maxent with the hyperparameters
% reg_path*lambda in an efficient manner.

reg_path = [1,0.95,0.90]; 



%% Extract the data and prepare it
% block0_values: 38 Features (see bottom of the script)
% block1_values: 7 Features (see bottom of the script)
% block2_values: 5 quantities (see bottom of the script)

% The features and quantities are listed at the bottom of the script,
% after the code for the solver and before the misc information.

% Read the data
A = [h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block0_values')',single(h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block1_values'))'];
data8 = h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block2_values')';

% Remove NaN values
ind_nan = isnan(A(:,1));
A = A(~ind_nan,:);
data8 = data8(~ind_nan,:);

% Separate data with fires and those with no fire.
ind_fire_yes = (data8(:,1) >= 1);

% Compute the number of background and presence-only data points
n0 = length(unique(data8(~ind_fire_yes,[2,4:5]),'rows'));
n1 = length(unique(data8(ind_fire_yes,[2,4:5]),'rows'));

% Normalize the features.
% Note: Background features are normalized together.
means_A = mean(A,1); A = A - means_A;
sumsq_A = sum(A.^2); A = A./sqrt(sumsq_A/(n0+n1)); % Normalization
b = A(ind_fire_yes,:);  % Presence data
A = A(~ind_fire_yes,:); % Background data

% Average over the features of the presence only data. We take the average
% w.r.t. the uniform distribution with n1 elements.
Ed = b'*ones(n1,1)/n1;

% Compute the smallest parameter for which the dual solution is zero.
lambda_est = norm(Ed - A'*(ones(n0,1)/n0),inf);

% Compute hyperparameters to be used
lambda = lambda_est*reg_path;

% Clear irrelevant arrays
clear data8 ind_fire_yes ind_nan



%% Parameters of the algorithm
% Dimensions
m = length(Ed); % Number of features
n = n0;         % Number of background samples
l_max = length(lambda); % Length of the regularization path

% Placeholders for solutions
sol_npdhg_w = zeros(m,l_max);
sol_npdhg_z = zeros(m,l_max);
sol_npdhg_p = zeros(n,l_max); sol_npdhg_p(:,1) = compute_p(A,sol_npdhg_z(:,1));

% Timings and Maximum number of iterations
time_npdhg_regular = 0;
time_npdhg_total = 0;
max_iter = 500;



%% Script for the nPDHG algorithm
disp(' ')
disp('Algorithm: The nPGHG method (with regular sequence + variable selection)')

% Compute the matrix norm for the nPDHG algorithm
tic
L12_sq = max(sum((A').^2));
time_L12 = toc;
disp(['Time for computing the maximum l2 norm of a row of A: ',num2str(time_L12), 's'])

% Regularization path
for i=2:1:l_max
    tic
    
    % Initialize the regularization hyperparameter and other parameters
    t = lambda(i); 
    theta = 0;
    sigma = 0.5/L12_sq;
    tau = 2;

    % Call the solver for this problem
    [sol_npdhg_w(:,i),sol_npdhg_z(:,i),num_iter_tot_reg] = ... 
        npdhg_l1_solver(sol_npdhg_w(:,i-1),sol_npdhg_z(:,i-1),t,A,tau,sigma,theta,Ed,max_iter);   
    sol_npdhg_p(:,i) = compute_p(A,sol_npdhg_z(:,i));
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
function p = compute_p(A,z)
% Compute a probability vector p from the formula
% p(j) = exp([A*z]_{j})/(sum_{j}exp([A*z]_{j}))
% for every j in {1,...,n} and some parameter z.

x = A*z; a = max(x);
w = exp(x-a);
p = w/sum(w);
end



%% Solver
function [sol_w,sol_z,num_iter_tot] = npdhg_l1_solver(w,z,lambda,A,tau,sigma,theta,Ed,max_iter)
% Nonlinear PDHG method for solving Maxent with 0.5*normsq{\cdot}.
% Input variables:
%   w = Array of dimension m x 1
%   z = Array of dimension m x 1
%   lambda = parameter > 0
%   A = An n x m matrix.
%   tau, sigma, gamma_h, m, d, max_iter = real numbers
%   Ed = Observed features of presence-only data. The features are averaged
%   w.r.t. a uniform distribution.

% Auxiliary variables
wminus = w;
p = A*z; p = exp(p-max(p));
p = p/sum(p);

% Counter for the iterations
num_iter_tot = 0;

% Iterations
for k=1:1:max_iter
    num_iter_tot = num_iter_tot + 1;
    
    % Update the primal variable and the probability
    zplus = (z + tau*(w + theta*(w-wminus)))/(1+tau);
    
    % Compute pplus
    temp = A*zplus; temp2 = max(temp);
    temp3 = exp(temp - temp2);
    pplus = temp3/sum(temp3);
    
    % Update the dual variable
    temp4 = A'*pplus;
    wplus = (w + sigma*(Ed-temp4));
    wplus = sign(wplus).*max(0,abs(wplus) - lambda*sigma);
    
    % Check for convergence
    if((k >= 100) && (mod(k,20) == 0))
        disp(norm(wplus-w)/norm(w))
        disp(norm(pplus-p,1))
        disp('---')
        if((norm(wplus-w) < (1e-03)*norm(w)) && (norm(pplus-p,1) < (1e-04)))
            break
        end
    end
    
    % Increment
    z = zplus;
    wminus = w; w = wplus;
    p = pplus;
    theta = 1/sqrt(1+tau); tau = theta*tau; sigma = sigma/theta;
end

% Final solutions
sol_z = zplus;
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