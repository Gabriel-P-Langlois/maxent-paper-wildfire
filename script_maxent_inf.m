%% Script written by Gabriel Provencher Langlois
% This script fits an \ell_{\inf}-regularized Maxent model on the fire data
% set provided by Jatan Buch using the either the nPDHG algorithm developed
% by Gabriel P. Langlois, the FISTA algorithm (Beck and Teboulle, 2009)
% or the coordinate descent method of Cortes and al. (2015)

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. 

% Run the script from the project directory where ./data is located.


%% Options for the script
% Options for new run and using quadratic features
new_run = false;

% Option to output the result at each iteration if desired
display_output = false;

% Option to save the results at the end if desired
save_results = false;

% Specify which algorithm to use. Note: Only one should be specified.
use_fista = true;
use_npdhg = false;

% Initialize the structure of the regularization path
reg_path = [1:-0.01:0.9,...
    0.895:-0.005:0.30,...
    0.2975:-0.0025:0.05];

% Tolerance for the optimality condition (for all methods)
tol = 1e-5;

% Maximum number of iterations in each algorithm before stopping
max_iter = 40000;


%% Data extraction
% amat_annual:      Design matrix
% pprior:           Prior distribution
% pempirical:       Empirical distribution (presence data)
% Ed:               Observed features (presence data)
% n0:               Number of background data points
% n1:               Number of presence data points
% name_features:    Name of the features
% idx_features:     Indices associated to the features
% ind_nan_months:     Indices of grid cells that are not used.
% groups:           Groupings of the features for the group lasso

% Note: We are averaging the temporal data individually over each month 
% (i.e., spatial data set over the months of January through December).

if(new_run)
    % Read the data, which is assumed to be stored locally (not on github).
    [amat_annual,pprior,pempirical,Ed,n0,n1,name_features,...
        ind_nan_months,groups] = prepare_wildfire_data;
    m = length(Ed);     % Number of features
end


%% Construct regularization path
% Sequence of hyperparameters, starting from smallest value /w dual = 0

lambda_est = norm(Ed - amat_annual'*pprior,1);
lambda = lambda_est*reg_path;
l_max = length(lambda);


%% Placeholder solutions and quantities for timings
sol_w = zeros(m,l_max);
sol_p = zeros(n0+n1,l_max); sol_p(:,1) = pprior;


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
%        disp(['Iteration ',num2str(i),'/',num2str(l_max)])

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
%        disp(['Iteration ',num2str(i),'/',num2str(l_max)])

        % Call the nPDHG solver
        theta = 0; tau = 2; sigma = 0.5/L12_sq;
        u_in = log(sol_p(:,i-1)./pprior);

        [sol_w(:,i),sol_p(:,i),num_iter_tot_reg] = ... 
            npdhg_solver_inf(sol_w(:,i-1),pprior,...
            lambda(i),amat_annual,tau,sigma,theta,Ed,...
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
% Save solutions and the regularization path to the data subdirectory
% Note: This assumes you are working from the project directory where
% the data folder is located.
if(save_results)
    save(strjoin(["data/generated_data/reg_path_linf",...
        ",min_path=",num2str(reg_path(end)),'.mat'],''),'lambda')
    
    % Note: Reshaped as a two-dimensional arrays of points (entries x
    % length_path). If the format (entries_per_month x months x
    % length_path) is desired, then uncomment the line just before the end
    % statement in the script reshape_probability
    reshaped_sol_p = reshape_probability(sol_p,ind_nan_months,length(reg_path));
    save(strjoin(["data/generated_data/p_sol_linf",...
        ",min_path=",num2str(reg_path(end)),'.mat'],''),'reshaped_sol_p')
    
    save(strjoin(["data/generated_data/w_sol_linf",...
        ",min_path=",num2str(reg_path(end)),'.mat'],''),'sol_w')
end


%% Postprocessing
% ind = 120;
% disp(lambda(ind)/lambda(1))
% 
% Emodel = (sol_p(:,ind).'*amat_annual).';
% deviation = Emodel - Ed;
% disp(sum(abs(deviation))); 
% disp(lambda(ind))


%% END