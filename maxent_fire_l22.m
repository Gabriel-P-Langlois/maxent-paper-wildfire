%% Script written by Gabriel Provencher Langlois
% This script performs l22-regularized Maxent on the fire data set
% provided by JB using the nPDHG algorithm developed by GPL and JD.

% The nPDHG algorithm computes probabilities indirectly using the 
% parametrization
% p(j) = pprior(j)*exp(<z,Phi(j)>)/(sum_{j}pprior(j)*exp(<z,Phi(j)>))
% for some vector z. This parametrization is motivated from the optimality
% conditions of regularized-Maxent.



%% Input
% Regularization path. Entries are numbers that multiply lambda_est.
% This script solves l22-regularized Maxent with the hyperparameters
% reg_path*lambda in an efficient manner.

reg_path = [1000,100,50,25]; 
%reg_path = [1000,100,50,25,10,5,1,0.5,0.25,0.1,0.05,0.01]; % 12 Entries. 



%% Extract the data and prepare it
% block0_values: 38 Features (see bottom of the script)
% block1_values: 7 Features (see bottom of the script)
% block2_values: 5 quantities (see bottom of the script)

% The features and quantities are listed at the bottom of the script,
% after the code for the solver and before the misc information.

% Read the data
% Note: The design matrix is an (n x m) matrix.
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
b = A(ind_fire_yes,:);
A = A(~ind_fire_yes,:);

% Average over the features of the presence only data. We take the average
% w.r.t. the uniform distribution with n1 elements.
Ed = b'*ones(n1,1)/n1;

% Compute an estimate of the parameter (for scaling purposes).
% The prior distribution is uniform w.r.t. all background samples, i.e.,
% ones(n0,1)/n0;
lambda_est = norm(Ed - A'*(ones(n0,1)/n0));

% Compute hyperparameters to be used
lambda = lambda_est*reg_path;

% Clear irrelevant arrays
clear data8 ind_fire_yes ind_nan



%% Parameters of the algorithm
% Dimensions
m = length(Ed); % Number of features
n = n0;         % Number of background samples
l_max = length(lambda); % Length of the regularization path

% Strong convexity factor
gamma_g = 1;

% Placeholders for solutions
% Note: The first columns of these quantities are the initial values
% for the Maxent problem at lambda(1). The solutions are stored
% in the columns 2:l_max+1.
sol_npdhg_w = zeros(m,l_max+1);
sol_npdhg_z = zeros(m,l_max+1);
sol_npdhg_p = zeros(n,l_max+1); sol_npdhg_p(:,1) = compute_p(A,sol_npdhg_z(:,1));

% Timings and Maximum number of iterations
time_npdhg_regular = 0;
time_npdhg_total = 0;
max_iter = 2000;



%% Script for the nPDHG algorithm
disp(' ')
disp('Algorithm: The nPGHG method (with regular sequence)')
    
% Compute the matrix norm for the nPDHG algorithm
tic
L12_sq = max(sum((A').^2));
time_L12 = toc;
disp(['Time for computing the maximum l2 norm of a row of A: ',num2str(time_L12), 's'])

% Regularization path
for i=1:1:l_max
    tic
    
    % Initialize the regularization hyperparameter and other parameters
    t = lambda(i); 
    gamma_h = t;
    mu = 0.5*gamma_g*gamma_h/L12_sq;
    theta = 1 - mu*(sqrt(1 + 2/mu)-1);
    tau = (1-theta)/(gamma_g*theta);
    sigma = 1/(theta*tau*L12_sq);

    % Call the solver for this problem
    [sol_npdhg_w(:,i+1),sol_npdhg_z(:,i+1),num_iter_tot_reg] = ... 
        npdhg_l22_solver(sol_npdhg_w(:,i),sol_npdhg_z(:,i),t,A,tau,sigma,theta,Ed,max_iter);   
    sol_npdhg_p(:,i+1) = compute_p(A,sol_npdhg_z(:,i+1));
    time_npdhg_regular = toc;
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_npdhg_regular),' seconds.'])
    disp(['Relative l22 deviation from presence-only data: ',num2str(norm(Ed - A'*sol_npdhg_p(:,i+1))/norm(Ed))]) % Measure of how regularized the problem is.
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
function [sol_w,sol_z,num_iter_tot] = npdhg_l22_solver(w,z,lambda,A,tau,sigma,theta,Ed,max_iter)
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
factor2 = 1/(1+tau);
factor3 = 1/(1+lambda*sigma);

%p = A*z; p = exp(p-max(p));
%p = p/sum(p);

% Counter for the iterations
num_iter_tot = 0;

% Iterations
for k=1:1:max_iter
    num_iter_tot = num_iter_tot + 1;
    
    % Update the primal variable and the probability
    zplus = (z + tau*(w + theta*(w-wminus)))*factor2;
    
    % Compute pplus
    temp = A*zplus; temp2 = max(temp);
    temp3 = exp(temp - temp2);
    pplus = temp3/sum(temp3);
    
    % Update the dual variable
    temp4 = A'*pplus;
    wplus = factor3*(w + sigma*(Ed-temp4));
 
    % Check for convergence
    if((k >= 20) && (mod(k,10) == 0))
        if((norm(wplus-w) < (1e-04)*norm(w)))
            break
        end
    end
    
    % Increment
    z = zplus;
    %p = pplus;
    wminus = w; w = wplus;
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
    
    
    
%% Information from a previous run
% Run for lambda_max*[100,50,25,10,5,1,0.5,0.25,0.1,0.05];
% Algorithm: The nPGHG method (with regular sequence)
% Time for computing the maximum l2 norm of a row of A: 0.86978s
% mu = 0.011333
% theta = 0.86036
% tau = 0.16231
% sigma = 0.00043338
% Final relative l1-norm of p: 1.2852e-05
% Final relative l2-norm of w: 3.4218e-05
% Solution computed for lambda = 3.7452e+02. Number of iterations = 50.
% Total time elapsed = 19.2472 seconds.
% Relative l22 deviation from presence-only data: 0.96824
%  
% mu = 0.0056664
% theta = 0.89906
% tau = 0.11227
% sigma = 0.00059955
% Final relative l1-norm of p: 1.9041e-05
% Final relative l2-norm of w: 3.456e-05
% Solution computed for lambda = 1.8726e+02. Number of iterations = 60.
% Total time elapsed = 26.6261 seconds.
% Relative l22 deviation from presence-only data: 0.93735
%  
% mu = 0.0028332
% theta = 0.9275
% tau = 0.078162
% sigma = 0.00083479
% Final relative l1-norm of p: 2.0519e-05
% Final relative l2-norm of w: 6.5954e-05
% Solution computed for lambda = 9.3631e+01. Number of iterations = 80.
% Total time elapsed = 33.55 seconds.
% Relative l22 deviation from presence-only data: 0.88196
%  
% mu = 0.0011333
% theta = 0.95351
% tau = 0.048755
% sigma = 0.0013018
% Final relative l1-norm of p: 2.2952e-05
% Final relative l2-norm of w: 8.6162e-05
% Solution computed for lambda = 3.7452e+01. Number of iterations = 120.
% Total time elapsed = 45.1867 seconds.
% Relative l22 deviation from presence-only data: 0.75455
%  
% mu = 0.00056664
% theta = 0.9669
% tau = 0.034236
% sigma = 0.0018282
% Final relative l1-norm of p: 6.5516e-05
% Final relative l2-norm of w: 8.3861e-05
% Solution computed for lambda = 1.8726e+01. Number of iterations = 140.
% Total time elapsed = 54.1626 seconds.
% Relative l22 deviation from presence-only data: 0.62073
%  
% mu = 0.00011333
% theta = 0.98506
% tau = 0.015169
% sigma = 0.0040502
% Final relative l1-norm of p: 4.2937e-05
% Final relative l2-norm of w: 8.7539e-05
% Solution computed for lambda = 3.7452e+00. Number of iterations = 340.
% Total time elapsed = 132.9881 seconds.
% Relative l22 deviation from presence-only data: 0.31722
%  
% mu = 5.6664e-05
% theta = 0.98941
% tau = 0.010702
% sigma = 0.0057152
% Final relative l1-norm of p: 4.7114e-05
% Final relative l2-norm of w: 9.7839e-05
% Solution computed for lambda = 1.8726e+00. Number of iterations = 370.
% Total time elapsed = 147.0102 seconds.
% Relative l22 deviation from presence-only data: 0.22138
%  
% mu = 2.8332e-05
% theta = 0.9925
% tau = 0.0075559
% sigma = 0.0080699
% Final relative l1-norm of p: 4.0745e-05
% Final relative l2-norm of w: 9.2132e-05
% Solution computed for lambda = 9.3631e-01. Number of iterations = 490.
% Total time elapsed = 190.0012 seconds.
% Relative l22 deviation from presence-only data: 0.14794
%  
% mu = 1.1333e-05
% theta = 0.99525
% tau = 0.0047722
% sigma = 0.012742
% Final relative l1-norm of p: 2.9594e-05
% Final relative l2-norm of w: 9.9686e-05
% Solution computed for lambda = 3.7452e-01. Number of iterations = 760.
% Total time elapsed = 289.5863 seconds.
% Relative l22 deviation from presence-only data: 0.082223
%  
% mu = 5.6664e-06
% theta = 0.99664
% tau = 0.0033721
% sigma = 0.018008
% Final relative l1-norm of p: 3.1065e-05
% Final relative l2-norm of w: 9.6555e-05
% Solution computed for lambda = 1.8726e-01. Number of iterations = 880.
% Total time elapsed = 317.1926 seconds.
% Relative l22 deviation from presence-only data: 0.050909
%  
% Total time elapsed for the nPDHG method = 1256.421 seconds.


%% 
% Algorithm: The nPGHG method (with regular sequence)
% Time for computing the maximum l2 norm of a row of A: 1.1778s
% Solution computed for lambda = 3.7452e+03. Number of iterations = 20.
% Total time elapsed = 11.9721 seconds.
% Relative l22 deviation from presence-only data: 0.99818
%  
% Solution computed for lambda = 3.7452e+02. Number of iterations = 40.
% Total time elapsed = 19.521 seconds.
% Relative l22 deviation from presence-only data: 0.96851
%  
% Solution computed for lambda = 1.8726e+02. Number of iterations = 50.
% Total time elapsed = 22.3153 seconds.
% Relative l22 deviation from presence-only data: 0.93772
%  
% Solution computed for lambda = 9.3631e+01. Number of iterations = 70.
% Total time elapsed = 29.1645 seconds.
% Relative l22 deviation from presence-only data: 0.88231
%  
% Solution computed for lambda = 3.7452e+01. Number of iterations = 90.
% Total time elapsed = 37.2427 seconds.
% Relative l22 deviation from presence-only data: 0.75513
%  
% Solution computed for lambda = 1.8726e+01. Number of iterations = 120.
% Total time elapsed = 52.7771 seconds.
% Relative l22 deviation from presence-only data: 0.61989
%  
% Solution computed for lambda = 3.7452e+00. Number of iterations = 290.
% Total time elapsed = 122.7489 seconds.
% Relative l22 deviation from presence-only data: 0.3165
%  
% Solution computed for lambda = 1.8726e+00. Number of iterations = 300.
% Total time elapsed = 133.663 seconds.
% Relative l22 deviation from presence-only data: 0.21992
%  
% Solution computed for lambda = 9.3631e-01. Number of iterations = 360.
% Total time elapsed = 145.4158 seconds.
% Relative l22 deviation from presence-only data: 0.14713
%  
% Solution computed for lambda = 3.7452e-01. Number of iterations = 520.
% Total time elapsed = 201.0798 seconds.
% Relative l22 deviation from presence-only data: 0.080283
%  
% Solution computed for lambda = 1.8726e-01. Number of iterations = 510.
% Total time elapsed = 196.933 seconds.
% Relative l22 deviation from presence-only data: 0.049397
%  
% Solution computed for lambda = 3.7452e-02. Number of iterations = 1140.
% Total time elapsed = 461.4906 seconds.
% Relative l22 deviation from presence-only data: 0.016159

% Total time: 1435.5017