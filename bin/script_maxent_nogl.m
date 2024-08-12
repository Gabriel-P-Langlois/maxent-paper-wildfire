%% Script written by Gabriel Provencher Langlois
% This script fits an non over-lapping group lasso regularized Maxent model 
% on the fire data set provided by Jatan Buch using either 
% the nPDHG algorithm developed by Langlois, Buch and Darbon (2024)
% or the FISTA algorithm by Beck and Teboulle, 2009.

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. See the manuscript
% ``Efficient first-order algorithms for large-scale, non-smooth maximum 
% entropy models with application to wildfire science" for further
% information (a preprint is available on
% (https://arxiv.org/pdf/2403.06816).

% NOTE 1: Run the script from the project directory.
% NOTE 2: This script requires the data file 
%   clim_fire_freq_12km_w2020_data.h5  that is too large
%   to be hosted on the github website. It can be found on Zenoto at
%   https://zenodo.org/records/7277980
%   Make sure it is located somewhere on the MATLAB path.


%% Options for the script
% Specify which algorithm to use. Note: Only one should be specified.
use_npdhg = true;
use_fista = false;

display_output = true;  % Print some information about the iterations
                        % of the algorithm if desired.
save_results = true;    % Save the results at the end if desired


%% Parameters
% Initialize the structure of the regularization path
reg_path = [1:-0.01:0.9,0.895:-0.005:0.30,0.2975:-0.0025:0.05];

% Tolerance for the optimality condition (for all methods)
tol = 1e-5;

% Maximum number of iterations in each algorithm before stopping
max_iter = 40000;


%% Data extraction
% Read the data, which is assumed to be stored locally.
% Note: The data file clim_fire_freq_12km_w2020_data.h5 is too large
% to be hosted on the github website. It can be found on Zenoto at
% https://zenodo.org/records/7277980
[mat_annual,pprior,Ed,n0,n1,...
    name_features,ind_nan_months,groups] = prepare_wildfire_data;


%% Construct regularization path for the non-overlapping group lasso
lambda_est = 0;
for i=1:1:length(groups)
    ind = groups{2,i};
    lambda_est = max(lambda_est,norm(Ed(ind) - mat_annual(:,ind)'*pprior,2)...
        /sqrt(length(ind)));
end
lambda = lambda_est*reg_path;
l_max = length(lambda);


%% Placeholder solutions and quantities for timings
p = length(Ed);
sol_w = zeros(p,l_max);
sol_p = zeros(n0+n1,l_max); sol_p(:,1) = pprior;


%% FIT THE NON-OVERLAPPING GROUP LASSO REGULARIZED MAXENT MODEL VIA NPDHG
if(use_npdhg)
    disp(' ')
    disp('The nPGHG methodfor nogl-regularized Maxent')

    time_total = 0;
    time_iter = 0; 
    L12_sq = max(sum((mat_annual.').^2));
    
    for i=2:1:l_max
        tic
        % Call the nPDHG solver
        theta = 0; tau = 2; sigma = 0.5/L12_sq;
        [sol_w(:,i),sol_p(:,i),num_iter_tot_reg] = ... 
            npdhg_solver_nogl(sol_w(:,i-1),pprior,...
            lambda(i),groups,mat_annual,tau,sigma,theta,Ed,...
            max_iter,tol);   

        time_iter = toc;
        time_total = time_total + time_iter;

        % If enabled, the code below will print info about the iteration
        if(display_output)
            disp(['Solution computed for lambda = ', ...
                num2str(lambda(i),'%.4e'), ...
                '. Number of primal-dual steps = ', ...
                num2str(num_iter_tot_reg), '.'])
            disp(['Time elapsed for solving the Maxent problem = ',...
                num2str(time_iter),' seconds.'])
            disp(' ')
        end
    end
    disp('----------')
    disp(['Total time elasped for constructing the regularization path' ...
        ' with the nPDHG algorithm: ',num2str(time_total), ' seconds.'])
    disp('----------')
end


%% FIT THE NON-OVERLAPPING GROUP LASSO REGULARIZED MAXENT MODEL VIA THE FISTA ALGORITHM
if(use_fista)
    disp(' ')
    disp('The FISTA method for nogl-regularized Maxent')
    
    time_total = 0;
    time_iter = 0; 
    L22 = svds(double(mat_annual),1);

    for i=2:1:l_max
        tic
        % Call the FISTA solver
        tau = 1/L22;
        mu = 0;
        q = 0;

        [sol_w(:,i),sol_p(:,i),num_iter_tot_reg] = ... 
            fista_solver_nogl(sol_w(:,i-1),pprior,...
            lambda(i),groups,mat_annual,tau,mu,q,Ed,...
            max_iter,tol);   

        time_iter = toc;
        time_total = time_total + time_iter;

        % If enabled, the code below will print info about the iteration
        if(display_output)
            disp(['Solution computed for lambda = ', ...
                num2str(lambda(i),'%.4e'), ...
                '. Number of primal-dual steps = ', ...
                num2str(num_iter_tot_reg), '.'])
            disp(['Time elapsed for solving the Maxent problem = ',...
                num2str(time_iter),' seconds.'])
            disp(' ')
        end
    end
    disp('----------')
    disp(['Total time elasped for constructing the regularization path' ...
        ' with the FISTA algorithm: ',num2str(time_total), ' seconds.'])
    disp('----------')
end


%% Save results, if enabled.
if(save_results)
    % Save the regularization path
    save(strjoin(["data/generated_data/reg_path_nogl,min_path=",...
        num2str(reg_path(end)),...
        '.mat'],''),'lambda')

    % Append the probability vector with the missing nan values
    % that were discarded in the prepare_wildfire_data.m function.
    % This is required to use the python script prob_plot_script.py
    reshaped_sol_p = ...
        reshape_probability(sol_p,ind_nan_months,length(reg_path));
    save(strjoin(["data/generated_data/p_sol_nogl,min_path=",...
        num2str(reg_path(end)),...
        '.mat'],''),'reshaped_sol_p')
    
    % Save the coefficients w characterizing the exponential family
    save(strjoin(["data/generated_data/w_sol_nogl,min_path=",...
        num2str(reg_path(end)),...
        '.mat'],''),'sol_w')

    % Save the hyperparameter thresholds for which we identify a new group
    % NOTE: Use lambda(ind_threshold_groups) to find the lambdas at which a
    % new group enters.
    ind_threshold_groups = identify_group_thresholds(sol_w,lambda,groups);
    save(strjoin(["data/generated_data/threshold_vals_nogl,min_path=",...
        num2str(reg_path(end)),...
        '.mat'],''),'ind_threshold_groups')
end
%% END