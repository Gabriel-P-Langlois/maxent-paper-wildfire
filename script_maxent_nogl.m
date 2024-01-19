%% Script written by Gabriel Provencher Langlois
% This script fits an non-overllaping group lasso regularized Maxent model
% on the fire data set provided by Jatan Buch using either the 
% nPDHG algorithm developed by Gabriel P. Langlois or
% the FISTA algorithm (Beck and Teboulle, 2009).

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. 

% Run the script from the project directory where ./data is located.


%% Notes


%% Options for the script
% Options for new run and using quadratic features
new_run = true;

% Option to output the result at each iteration if desired
display_output = false;

% Option to save the results at the end if desired
save_results = false;

% Specify which algorithm to use. Note: Only one should be specified.
use_fista = false;
use_npdhg = true;

% Initialize the structure of the regularization path
reg_path = [1:-0.01:0.5,...
    0.495:-0.005:0.10];

% Tolerance for the optimality condition (for all methods)
tol = 1e-5;

% Maximum number of iterations in each algorithm before stopping
max_iter = 20000;


%% Data extraction
% amat_annual:      Design matrix
% pprior:           Prior distribution
% pempirical:       Empirical distribution (presence data)
% Ed:               Observed features (presence data)
% n0:               Number of background data points
% n1:               Number of presence data points
% name_features:    Name of the features
% idx_features:     Indices associated to the features
% ind_nan_mths:     Indices of grid cells that are not used
% groups:           Groupings of the features for the group lasso

if(new_run)
    % Read the data, which is assumed to be stored locally (not on github).
    [amat_annual,pprior,pempirical,Ed,n0,n1,name_features,...
        ind_nan_mths,groups] = prepare_wildfire_data;

    % Total number of features
    m = length(Ed);     
end


%% Construct regularization path for the elastic net method
lambda_est = 0;
for i=1:1:length(groups)
    ind = groups{2,i};
    lambda_est = max(lambda_est,norm(Ed(ind) - amat_annual(:,ind)'*pprior,2)...
        /sqrt(length(ind)));
end
lambda = lambda_est*reg_path;
l_max = length(lambda);


%% Placeholder solutions and quantities for timings
sol_w = zeros(m,l_max);
sol_p = zeros(n0+n1,l_max); sol_p(:,1) = pprior;


%% FIT AN ELASTIC NET REGULARIZED MAXENT MODEL VIA THE FISTA ALGORITHM
if(use_fista)
    disp(' ')
    disp('The FISTA method \w variable selection for nogl-regularized Maxent')
    
    time_total = 0;
    time_iter = 0; 
    
    L22 = svds(double(amat_annual),1);

    for i=2:1:l_max
        tic

        % Display percentage of zero coefficients
        disp(['Iteration ',num2str(i),'/',num2str(l_max)])
        
        % Call the FISTA solver
        tau = 1/L22;
        mu = 0;
        q = 0;

        [sol_w(:,i),sol_p(:,i),num_iter_tot_reg] = ... 
            fista_solver_nogl(sol_w(:,i-1),pprior,...
            lambda(i),groups,amat_annual,tau,mu,q,Ed,...
            max_iter,tol);   

        time_iter = toc;
        time_total = time_total + time_iter;

        % If enabled, the code below will print info about the iteration
        if(display_output)
            disp(['Solution computed for lambda = ', num2str(lambda(i),'%.4e'), '. Number of primal-dual steps = ', num2str(num_iter_tot_reg), '.'])
            disp(['Time elapsed for solving the Maxent problem = ',...
                num2str(time_iter),' seconds.'])
            disp(' ')
        end
    end
    disp('----------')
    disp(['Total time elasped for constructing the regularization path with the FISTA algorithm: ',num2str(time_total), ' seconds.'])
    disp('----------')
end


%% FIT AN ELASTIC NET REGULARIZED MAXENT MODEL VIA NPDHG
if(use_npdhg)
    disp(' ')
    disp('The nPGHG method \w variable selection for nogl-regularized Maxent')

    time_total = 0;
    time_iter = 0; 

    L12_sq = max(sum((amat_annual.').^2));
    
    for i=2:1:l_max
        tic

        % Display percentage of zero coefficients
        disp(['Iteration ',num2str(i),'/',num2str(l_max)])
        
        % Call the nPDHG solver
        theta = 0; tau = 2; sigma = 0.5/L12_sq;
        [sol_w(:,i),sol_p(:,i),num_iter_tot_reg] = ... 
            npdhg_solver_nogl(sol_w(:,i-1),pprior,...
            lambda(i),groups,amat_annual,tau,sigma,theta,Ed,...
            max_iter,tol);   

        time_iter = toc;
        time_total = time_total + time_iter;

        % If enabled, the code below will print info about the iteration
        if(display_output)
            disp(['Solution computed for lambda = ', num2str(lambda(i),'%.4e'), '. Number of primal-dual steps = ', num2str(num_iter_tot_reg), '.'])
            disp(['Time elapsed for solving the Maxent problem = ',...
                num2str(time_iter),' seconds.'])
            disp(' ')
        end
    end
    disp('----------')
    disp(['Total time elasped for constructing the regularization path with the nPDHG algorithm: ',num2str(time_total), ' seconds.'])
    disp('----------')
end


%% Save results
% Save solutions and the regularization path to the data subdirectory,
% which assumes you are working from the project directory.
if(save_results)
    save(strjoin(["data/generated_data/reg_path_nogl,min_path=",...
        num2str(reg_path(end)),'.mat'],''),'lambda')
    
    save(strjoin(["data/generated_data/p_sol_nogl,min_path=",...
        num2str(reg_path(end)),'.mat'],''),'sol_p')
    
    save(strjoin(["data/generated_data/w_sol_nogl,min_path=",...
        num2str(reg_path(end)),'.mat'],''),'sol_w')

    % Save the hyperparameter thresholds for which we identify a new group
    % NOTE: Use lambda(ind_threshold_groups) to find the lambdas at which a
    % new group enters.
    ind_threshold_groups = identify_group_thresholds(sol_w,lambda,groups);
    save(strjoin(["data/generated_data/threshold_vals_alpha=",num2str(alpha),...
       ",min_path=",num2str(reg_path(end)),...
       ",quad_features=",num2str(use_quadratic_features),'.mat'],''),...
    'ind_threshold_groups')
end


%% Postprocessing
% Identify group thresholds
display_features_results(sol_w,lambda,name_features,groups)

% Plot the regularization path
print_regularization_path(sol_w,lambda,groups);


%% END