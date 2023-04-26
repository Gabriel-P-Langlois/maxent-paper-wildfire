%% Script written by Gabriel Provencher Langlois
% This script performs l1-regularized Maxent on the fire data set
% provided by JB using the nPDHG algorithm developed by GPL.

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. Here, we are averaging
% the temporal data individually over each month (i.e., spatial data set
% over the months of January through December).


%% Input
% Regularization path stored as an array. The entries must be positive
% and decreasing numbers starting from one.

reg_path = [1:-0.01:0.75,0.745:-0.005:0.50];%,...
     %0.4975:-0.0025:0.35,0.349:-0.001:0.20];

% reg_path = [1:-0.01:0.75,0.745:-0.005:0.50,...
%      0.4925:-0.0025:0.35,0.349:-0.001:0.20];

% reg_path = [1,0.96,0.95:-0.01:0.75,0.745:-0.005:0.50,...
%      0.4925:-0.0025:0.35,0.349:-0.001:0.20,0.19995:-0.0005:0.15];

% reg_path = [1,0.96,0.95:-0.01:0.75,0.745:-0.005:0.50,...
%     0.4925:-0.0025:0.35,0.349:-0.001:0.20,0.19995:-0.0005:0.125,...
%     0.12475:-0.00025:0.05];

% Tolerance for the optimality condition.
tol = 1e-04;             


%% Data extraction
[amat_annual,pprior,pempirical,Ed,n0,n1,name_features,idx_features] = process_augment_wildfire_data;


%% Compute hyperparameters
% Smallest hyperparameter for which the dual solution is zero.
lambda_est = norm(Ed - amat_annual'*pprior,inf);

% Sequence of hyperparameters
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
max_iter = 4000;

% Other quantities
theta = 0; tau = 2;


%% Script for the nPDHG algorithm
disp(' ')
disp('Algorithm: The nPGHG method (with regular sequence + variable selection)')

% Regularization path
time_L12 = 0;
for i=2:1:l_max
    tic
    disp(['Iteration ',num2str(i),'/',num2str(l_max)])
    t = lambda(i); 
    
    % Variable selection for l1-Maxent
    % Based on unpublished results.
    alpha = t/lambda(i-1);
    s = alpha*sol_npdhg_p(:,i-1) + (1-alpha)*pempirical;
    lhs = abs(Ed - amat_annual.'*sol_npdhg_p(:,i-1));    
    rhs = lambda(i-1) - ones(m,1)*sqrt(2*(s.'*log(s./sol_npdhg_p(:,i-1))))/alpha;
    ind = (lhs >= rhs);
    
    % Display coefficients found to be zero.
    disp(['Percentage of coefficients found to be zero: ',num2str(100-100*sum(ind)/m)])
    
    % Initialize the regularization hyperparameter and other parameters
    tic
    L12_sq = max(sum((amat_annual(:,ind).').^2));
    time_L12 = time_L12 + toc;
    sigma = 0.5/L12_sq;

    % Call the solver for this problem and compute the resultant
    % probability distribution.    

    [sol_npdhg_w(ind,i),sol_npdhg_p(:,i),num_iter_tot_reg] = ... 
        solver_l1_npdhg(sol_npdhg_w(ind,i-1),log(s./pprior),t,amat_annual(:,ind),tau,sigma,theta,Ed(ind),max_iter,tol);   
    time_npdhg_regular = toc;
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of primal-dual steps = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_npdhg_regular),' seconds.'])
    disp(' ')
    time_npdhg_total = time_npdhg_total + time_npdhg_regular;
end
disp(['Total time elapsed for the nPDHG method = ',num2str(time_npdhg_total + time_L12),' seconds.'])


%% Postprocessing
idx_lambda_found = l1_wildfire_postprocessing(sol_npdhg_w,reg_path,name_features);


%% (WIP) Computing the probability map
% Convert the probability vector into an h5 file that Jatan can process
% using his Python code. Jatan has the function for that.
%: Must be of double type!

h5create('my_example_file.h5', '/dataset1', size(sol_npdhg_p(:,end)));
h5write('my_example_file.h5', '/dataset1', sol_npdhg_p(:,end));

% Visualize the probability vector

%% END OF THE SCRIPT