%% Script written by Gabriel Provencher Langlois
% This script performs l1-regularized Maxent on the fire data set
% provided by JB using the coordinate descent algorithm in the
% Cortes et al. (2015) paper: "Structural Maxent Models".

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. Here, we are averaging
% the temporal data individually over each month (i.e., spatial data set
% over the months of January through December).


%% Notes
% Use optimality conditions of each subproblems? 
% In particular of the dual problem?


%% Input
% Regularization path stored as an array. The entries must be positive
% and decreasing numbers starting from one.
% reg_path = [1,0.96,0.95:-0.01:0.75,0.745:-0.005:0.50,...
%      0.4925:-0.0025:0.35];

reg_path = [1,0.96,0.95:-0.01:0.75,0.745:-0.005:0.50,...
     0.4925:-0.0025:0.35,0.349:-0.001:0.20,0.19995:-0.0005:0.15];

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
m = length(Ed);         % Number of features
l_max = length(lambda); % Length of the regularization path

% Placeholders for solutions
sol_npdhg_w = single(zeros(m,l_max));
sol_npdhg_p = single(zeros(n0+n1,l_max)); sol_npdhg_p(:,1) = pprior;

% Timings and Maximum number of iterations
time_cd_regular = 0;
time_cd_total = 0;
max_iter = 20000;


%% Script for the coordinate descent algorithm
disp(' ')
disp('Algorithm: Coordinate Descent (Cortes et al. (2015)')

% Regularization path
for i=2:1:l_max
    tic
    disp(['Iteration ',num2str(i),'/',num2str(l_max)])
    t = lambda(i); 
    
    % Variable selection
    alpha = t/lambda(i-1);
    s = alpha*sol_npdhg_p(:,i-1) + (1-alpha)*pempirical;
    lhs = abs(Ed - amat_annual.'*sol_npdhg_p(:,i-1));   
    rhs = lambda(i-1) - ones(m,1)*sqrt(2*(s.'*log(s./sol_npdhg_p(:,i-1))))/alpha;
    ind = (lhs >= rhs);         % Indices that are non-zero.
    
    % Display coefficients found to be zero.
    disp(['Percentage of coefficients found to be zero: ',num2str(100-100*sum(ind)/m)])

    % Call the solver for this problem and compute the resultant
    % probability distribution.    
    [sol_npdhg_w(ind,i),sol_npdhg_p(:,i),num_iter_tot_reg] = ... 
        solver_l1_cd(sol_npdhg_w(ind,i-1),s,t,amat_annual(:,ind),Ed(ind),max_iter,tol);   
    time_cd_regular = toc;
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_cd_regular),' seconds.'])
    disp(' ')
    time_cd_total = time_cd_total + time_cd_regular;
end
disp(['Total time elapsed for the CD method = ',num2str(time_cd_total),' seconds.'])


%% Postprocessing
idx_lambda_found = l1_wildfire_postprocessing(sol_npdhg_w,reg_path,name_features);

%% END OF THE SCRIPT