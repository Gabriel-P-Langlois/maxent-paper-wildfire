%% Script written by Gabriel Provencher Langlois
% This script fits an \ell_{\inf}-regularized Maxent model on the fire data
% set provided by Jatan Buch using the either the nPDHG algorithm developed
% by Gabriel P. Langlois, the FISTA algorithm (Beck and Teboulle, 2009)
% or the coordinate descent method of Cortes and al. (2015)

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. 


%% Notes


%% Options for the script
% Only modify the values below and not anywhere else in the script

% Check for a new run. Set to 1 if you are doing a new run, 0 otherwise.
% If set to 0, the script will not attempt to load the data
new_run = 1;

% Option to output the result at each iteration if desired
display_output = false;

% Option to save the results at the end if desired
save_results = false;

% Specify which algorithm to use. Note: Only one should be specified.
use_fista = false;
use_npdhg = true;

% Option to use quadratic features
use_quadratic_features = false;

% Initialize structure of the regularization path
npts_path = 50;
min_val_path = 0.50;

reg_path = linspace(1,min_val_path,npts_path);

% Tolerance for the optimality condition (for all methods)
tol = 1e-8;

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
% ind_nan_mths:     Indices of grid cells that are not used.

% Note: We are averaging the temporal data individually over each month 
% (i.e., spatial data set over the months of January through December).

if(new_run)
    % Read the data. Since it's too large, it is stored locally
    % On GPL computer, the data is located at ./.. from the project
    % directory.
    [amat_annual,pprior,pempirical,Ed,n0,n1,name_features,idx_features,...
        ind_nan_mths] = prepare_wildfire_data(use_quadratic_features);
    m = length(Ed);     % Number of features
end


%% Construct regularization path
% Sequence of hyperparameters, starting from smallest value /w dual = 0

lambda_est = norm(Ed - amat_annual'*pprior,1);
lambda = lambda_est*reg_path;
l_max = length(lambda);


%% Placeholder solutions and quantities for timings
sol_w = single(zeros(m,l_max));
sol_p = single(zeros(n0+n1,l_max)); sol_p(:,1) = pprior;


%% FIT AN linf REGULARIZED MAXENT MODEL VIA THE FISTA ALGORITHM
if(use_fista)
    disp(' ')
    disp('The FISTA method \w variable selection for linf Maxent')
    disp(' ')
    
    time_total = 0;

    % Initialize parameters
    L22 = svds(double(amat_annual),1);
    
    for i=2:1:l_max
        tic
        disp(['Iteration ',num2str(i),'/',num2str(l_max)])

        % Call the FISTA solver
        tau = 1/L22;
        mu = 0;
        q = 0;

        [sol_w(:,i),sol_p(:,i),num_iter_tot_reg] = ... 
            fista_solver_inf(sol_w(:,i-1),pprior,...
            lambda(i),amat_annual,tau,mu,q,Ed,...
            max_iter,tol);   

        time_iter = toc;
        time_total = time_total + time_iter;

        if(display_output) % Display outcome
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


%% FIT AN linf REGULARIZED MAXENT MODEL VIA NPDHG
if(use_npdhg)
    disp(' ')
    disp('The nPGHG method \w variable selection for linf Maxent')
    disp(' ')

    time_total = 0;

    % Initialize parameters
    L12_sq = max(sum((amat_annual.').^2));

    for i=2:1:l_max
        tic
        disp(['Iteration ',num2str(i),'/',num2str(l_max)])

        % Call the nPDHG solver
        theta = 0; tau = 2; sigma = 0.5/L12_sq;
        u_in = log(sol_p(:,i-1)./pprior);

        [sol_w(:,i),sol_p(:,i),num_iter_tot_reg] = ... 
            npdhg_solver_inf(sol_w(:,i-1),u_in,...
            lambda(i),amat_annual,tau,sigma,theta,Ed,...
            max_iter,tol); 

        time_iter = toc;
        time_total = time_total + time_iter;

        if(display_output) % Display outcome
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
% Save solutions and the regularization path to the data subdirectory
% Note: This assumes you are working from the project directory where
% the data folder is located.
if(save_results)
    save(strjoin(["data/generated_data/reg_path_linf,npts=",...
        num2str(npts_path),",min_path=",num2str(min_val_path),...
        ",quad_features=",num2str(use_quadratic_features),'.mat'],''),'lambda')
    
    save(strjoin(["data/generated_data/p_sol_linf,npts=",...
        num2str(npts_path),",min_path=",num2str(min_val_path),...
        ",quad_features=",num2str(use_quadratic_features),'.mat'],''),'sol_p')
    
    save(strjoin(["data/generated_data/w_sol_linf,npts=",...
        num2str(npts_path),",min_path=",num2str(min_val_path)...
        ",quad_features=",num2str(use_quadratic_features),'.mat'],''),'sol_w')
    % save(strjoin["data/generated_data/name_features"],'name_features')
end



%% END