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
reg_path = [100,20,10,0.75,0.5,0.25,0.15,0.10,0.05,0.01];



%% Extract the data and prepare it
% Note: The features and quantities are listed at the bottom of the script,
% after the code for the solver and before the misc information.

% block0_values: 38 Features (see bottom of the script)
% block1_values: 7 Features (see bottom of the script)
% block2_values: 5 quantities (see bottom of the script)

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
val = Ed - A'*(ones(n,1)/n);
lambda_est = norm(Ed - A'*(ones(n,1)/n));

% Compute hyperparameters to be used
lambda = lambda_est*reg_path;

% Clear irrelevant arrays
clear data8 ind_fire_yes ind_nan



%% Parameters of the algorithm
% Dimensions
m = length(Ed); % Number of features
l_max = length(lambda); % Length of the regularization path

% Strong convexity factor
gamma_g = 1;

% Placeholders for solutions
% Note: The first columns of these quantities are the initial values
% for the Maxent problem at lambda(1). The solutions are stored
% in the columns 2:l_max+1.
sol_npdhg_w = zeros(m,l_max+1); sol_npdhg_w(:,1) = val/lambda(1);
sol_npdhg_z = zeros(m,l_max+1);
sol_npdhg_p = zeros(n,l_max+1); sol_npdhg_p(:,1) = ones(n,1)/n;

% Timings and Maximum number of iterations
time_npdhg_regular = 0;
time_npdhg_total = 0;
max_iter = 1000;



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

    % Call the solver for this problem and compute the resultant
    % probability distribution
    [sol_npdhg_w(:,i+1),sol_npdhg_z(:,i+1),sol_npdhg_p(:,i+1),Ed_minus_Ep,num_iter_tot_reg] = ... 
        npdhg_l22_solver(sol_npdhg_w(:,i),sol_npdhg_z(:,i),t,A,tau,sigma,theta,Ed,max_iter);   
    time_npdhg_regular = toc;
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_npdhg_regular),' seconds.'])
    disp(['Relative l22 deviation from presence-only data: ', num2str(norm(Ed_minus_Ep)/norm(Ed))])
    disp(' ')
    time_npdhg_total = time_npdhg_total + time_npdhg_regular;
end
disp(['Total time elapsed for the nPDHG method = ',num2str(time_npdhg_total + time_L12),' seconds.'])
% End of the script.



%% Solver
function [sol_w,sol_z,sol_p,Ed_minus_Ep,num_iter_tot] = npdhg_l22_solver(w,z,lambda,A,tau,sigma,theta,Ed,max_iter)
% Nonlinear PDHG method for solving Maxent with 0.5*normsq{\cdot}.
% Input variables:
%   w: m x 1 vector -- Weights of the gibbs distribution.
%   z: n x 1 vector -- Parameterization of the gibbs probability
%   distribution, where p(j) = pprior(j)e^{<z,Phi(j)>-C}.
%   lambda: Positive number -- Hyperparameter.
%   A: n x m matrix -- Matrix of features (m) for each grid point (n).
%   tau, sigma, gamma_h: Positive numbers -- Stepsize parameters.
%   Ed: m-dimensional vector -- Observed features of presence-only data. 
%   max_iter: Positive integer -- Maximum number of iterations.

% Auxiliary variables
wminus = w;
factor2 = 1/(1+tau);
factor3 = 1/(1+lambda*sigma);

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
    temp4 = Ed-temp4;
    wplus = factor3*(w + sigma*(temp4));
 
    % Check for convergence
    if((k >= 15) && (mod(k,5) == 0))
        if((norm(temp4 - lambda*wplus) < (1e-04)))
            break
        end
    end
    
    % Increment
    z = zplus;
    wminus = w; w = wplus;
end

% Final solutions
sol_w = wplus;
sol_z = zplus;
sol_p = pplus;
Ed_minus_Ep = temp4;
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