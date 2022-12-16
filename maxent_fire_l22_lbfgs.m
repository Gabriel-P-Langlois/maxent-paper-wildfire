%% Script written by Gabriel Provencher Langlois
% This script performs l22-regularized Maxent on the fire data set
% provided by JB using L-BFGS as implemented in MATLAB.

% The nPDHG algorithm computes probabilities indirectly using the 
% parametrization
% p(j) = pprior(j)*exp(<z,Phi(j)>)/(sum_{j}pprior(j)*exp(<z,Phi(j)>))
% for some vector z. This parametrization is motivated from the optimality
% conditions of regularized-Maxent.



%% Notes
% For the path [100,20,10,0.75,0.5,0.25,0.15,0.10,0.05,0.01,0.0075,0.005]:
% L-BFGS: ~ 56.8978 seconds with tol 1e-04 and double precision.

%% Input
% Regularization path. Entries are numbers that multiply lambda_est.
% This script solves l22-regularized Maxent with the hyperparameters
% reg_path*lambda in an efficient manner.
reg_path = [100,20,10,0.75,0.5,0.25,0.15,0.10,0.05,0.01,0.0075,0.005];


%% Extract the data and prepare it
% Note: The features and quantities are listed at the bottom of the script,
% after the code for the solver and before the misc information.

% block0_values: 38 Features (see bottom of the script)
% block1_values: 7 Features (see bottom of the script)
% block2_values: 5 quantities (see bottom of the script)

% Read the data
A = double([h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block0_values')',h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block1_values')']);
data8 = double(h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block2_values')');

% Remove NaN values
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

% Take the transpose of the matrix
A = A';

% Average over the features of the presence only data. We take the average
% w.r.t. the uniform distribution with n1 elements.
% Note: We can weigh the background vs presence only data differently.
pempirical = zeros(n,1); pempirical(ind_fire_yes) = 1/n1;
Ed = A*pempirical;

% Compute the smallest parameter for which the dual solution is zero.
% Note: The prior distribution is uniform w.r.t. to the background *AND* 
% presence samples. 
val = Ed - A*(ones(n,1)/n);
lambda_est = norm(Ed - A*(ones(n,1)/n));

% Compute hyperparameters to be used
lambda = lambda_est*reg_path;

% Clear irrelevant arrays
clear data8 ind_fire_yes ind_nan



%% Parameters of the algorithm
% Dimensions
m = length(Ed); % Number of features
l_max = length(lambda); % Length of the regularization path

% Placeholders for solutions
% Note: The first columns of these quantities are the initial values
% for the Maxent problem at lambda(1). The solutions are stored
% in the columns 2:l_max+1.

% Placeholders for solutions
% Note: The first columns of these quantities are the initial values
% for the Maxent problem at lambda(1). The solutions are stored
% in the columns 2:l_max+1.
sol_npdhg_w = zeros(m,l_max+1); sol_npdhg_w(:,1) = val/lambda(1);

% Timings and Maximum number of iterations
max_iter = 2000;
time_lbfgs = 0;
time_lbfgs_total = 0;


%% Script for the lBFGS algorithm
disp(' ')
disp('Algorithm: The lBFGS method (with regular sequence)')

% Regularization path
for i=1:1:l_max
    tic
    
    % Call the solver for this problem
    t = lambda(i);
    sol_npdhg_w(:,i+1) = lbfgs_solver(sol_npdhg_w(:,i),t,A,Ed);
    time_lbfgs = toc;
    Ed_minus_Ep = Ed - (A*compute_p(A,sol_npdhg_w(:,i+1)));
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '.'])
    disp(['Total time elapsed = ',num2str(time_lbfgs),' seconds.'])
    disp(['Relative l22 deviation from presence-only data: ', num2str(norm(Ed_minus_Ep)/norm(Ed))])
    disp(' ')
    time_lbfgs_total = time_lbfgs_total + time_lbfgs;
end
disp(['Total time elapsed for the l-BFGS method = ',num2str(time_lbfgs_total + time_lbfgs),' seconds.'])
% End of the script.



%% Auxiliary functions
function p = compute_p(A,w)
% Compute a probability vector p from the formula
% p(j) = exp([A*w]_{j})/(sum_{j}exp([A*w]_{j}))
% for every j in {1,...,n} and vector w.
% The matrix A is an m x n matrix.

x = (w.'*A).';
w = exp(x-max(w));
p = w/sum(w);
end



%% Solver
function sol_w = lbfgs_solver(w0,lambda,A,Ed)
% Nonlinear PDHG method for solving Maxent with 0.5*normsq{\cdot}.
% Input variables:
%   w = Array of dimension m x 1. This is the starting point.
%   lambda = Positive scalar.
%   A = An m x n matrix.
%   Ed = Observed features of presence-only data.

options = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'MaxIterations',400,'OptimalityTolerance',1e-04,'Display','off');
sol_w = fminunc(@l2maxent,w0,options);

    function [f,g] = l2maxent(w)
        % Note: The prior is assumed to be uniform.
        % Compute the objective function
        p = (w.'*A)'; a = max(p);
        p = exp(p-a); b = sum(p);
        log_term = a + log(b);
        f= log_term + (0.5*lambda)*sum((w.^2)) - Ed'*w;
       
        % Compute the gradient of the objective function
        g = (A*p)/b;
        g = g + lambda*w - Ed;
    end
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