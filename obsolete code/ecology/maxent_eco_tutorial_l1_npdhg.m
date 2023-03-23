%% Script written by Gabriel Provencher Langlois
% This script performs l1-regularized Maxent on the tutorial ecology data
% set provided by Phillips et al. using the nPDHG algorithm.


%% Input
reg_path = [1:-0.001:0.05];

% Tolerance for the optimality condition.
tol = 1e-04;  


%% Data extraction
% Read background samples
T_background = readtable('background.csv');
amat = T_background{:,4:end}; % Feature matrix
n0 = size(amat,1);

% Read presence-only data
T_presence = readtable('bradypus_swd.csv');
amat_presence = T_presence{:,4:end};
n1 = size(amat_presence,1);

% Concatenate the two
amat = [amat_presence;amat];
clear amat_presence T_background T_presence


%% Data preparation
% Create new features
% TBC.

% Scale the features
max_amat = max(amat);
min_amat = min(amat);
amat = (amat - min_amat)./...
    (max_amat-min_amat);

% Extract empirical features
pempirical = ones(n1,1)/n1;         % Assumes no duplicates
Ed = amat(1:n1,:).'*pempirical;

%% Algorithm preparation
% Define the prior
pprior = ones(n0+n1,1)/(n0+n1);

% Compute the smallest parameter for which the dual solution is zero.
lambda_est = norm(Ed - amat.'*pprior,inf);

% Compute hyperparameters to be used
lambda = lambda_est*reg_path;


%% Parameters of the algorithm
% Dimensions
m = length(Ed);          % Number of features
l_max = length(lambda);  % Length of the regularization path

% Placeholders for solutions
sol_npdhg_w = single(zeros(m,l_max));
sol_npdhg_p = single(zeros(n0+n1,l_max)); sol_npdhg_p(:,1) = pprior;

% Timings and Maximum number of iterations
time_npdhg_regular = 0;
time_npdhg_total = 0;
max_iter = 5000;

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
    disp(['Iteration ',num2str(i),'/',num2str(l_max)])
    t = lambda(i); 
    
    % Initialize the regularization hyperparameter and other parameters
    tic
    L12_sq = max(sum((amat.').^2));
    time_L12 = time_L12 + toc;
    sigma = 0.5/L12_sq;

    % Call the solver for this problem and compute the resultant
    % probability distribution.    
    [sol_npdhg_w(:,i),sol_npdhg_p(:,i),num_iter_tot_reg] = ... 
        npdhg_l1_solver(sol_npdhg_w(:,i-1),sol_npdhg_p(:,i-1),t,amat,tau,sigma,theta,Ed,max_iter,tol);   
    time_npdhg_regular = toc;
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of primal-dual steps = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_npdhg_regular),' seconds.'])
    disp(' ')
    time_npdhg_total = time_npdhg_total + time_npdhg_regular;
end
disp(['Total time elapsed for the nPDHG method = ',num2str(time_npdhg_total + time_L12),' seconds.'])


%% Postprocessing
% Example 1: Sparsity plot
sparsity_plot = zeros(l_max,1);
for j=1:1:l_max
    sparsity_plot(j) = sum((sol_npdhg_w(:,j)~= 0));
end
plot(reg_path,sparsity_plot,'*')
xlabel('\lambda/\lambda_{max}')
ylabel('Nb of non-zero coefficients')
set ( gca, 'xdir', 'reverse' )
title('Sparsity plot')


% % Example 2: Rank the features
% disp('Nonzero features discovered, in order in which they appeared: ')
% disp(' ')
% name_features = [h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block0_items')',h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block1_items')'].';
% nb_features = max(sparsity_plot);
% features_ranked = zeros(nb_features,1);
% for j=1:1:nb_features
%     i_target = find(sparsity_plot == j,1);
%     ind_target = find(sol_npdhg_w(:,i_target) ~= 0);
%     features_ranked(j,1) = setdiff(ind_target,features_ranked);
%     if(j < 10)
%         disp([num2str(j),'.   ',num2str(name_features(features_ranked(j,1)))])
%     else
%         disp([num2str(j),'.  ',num2str(name_features(features_ranked(j,1)))])
%     end
% end
% disp(' ')