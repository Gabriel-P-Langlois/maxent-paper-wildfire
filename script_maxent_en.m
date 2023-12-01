%% Script written by Gabriel Provencher Langlois
% This script fits an elastic net regularized Maxent model on the fire data
% set provided by Jatan Buch using either the nPDHG algorithm developed
% by Gabriel P. Langlois, the FISTA algorithm (Beck and Teboulle, 2009)
% or the coordinate descent method of Cortes and al. (2015)

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. 

% Run the script from the project directory where ./data is located.


%% Notes
% The regularization path (alpha = 0.95, with no quadratic features)
% reg_path = [1:-0.01:0.9,...
%     0.895:-0.005:0.30,...
%     0.2975:-0.0025:0.01];
% takes about six minutes to complete with the nPDHG algorithm on GPL's
% laptop.

% The regularization path (alpha = 0.35, with no quadratic features)
% reg_path = [1:-0.01:0.9,...
%     0.895:-0.005:0.30,...
%     0.2975:-0.0025:0.01,...
%     0.009:-0.001:0.001,...
%     0.0009:-0.00005:0.0005];
% takes about four minutes to complete with the nPDHG algorithm on GPL's
% laptop.


%% Options for the script
% Options for new run and using quadratic features
new_run = true;
use_quadratic_features = false;

% Option to output the result at each iteration if desired
display_output = false;

% Option to save the results at the end if desired
save_results = false;

% Specify which algorithm to use. Note: Only one should be specified.
use_fista = false;
use_cdescent = false;
use_npdhg = true;

% Elastic net parameter
alpha = 0.35;

% Initialize the structure of the regularization path
reg_path = [1:-0.01:0.9,...
    0.895:-0.005:0.30,...
    0.2975:-0.0025:0.01,...
    0.009:-0.001:0.001,...
    0.0009:-0.00005:0.0005];

% Threshold for using linear vs sublinear (sublinear if alpha > threshold)
threshold = 0.45;

% Tolerance for the optimality condition (for all methods)
tol = 1e-5;

% Maximum number of iterations in each algorithm before stopping
max_iters = 20000;


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
% groups:           Groupings of the features for the group lasso

% Note: We are averaging the temporal data individually over each month 
% (i.e., spatial data set over the months of January through December).

if(new_run)
    % Read the data, which is assumed to be stored locally (not on github).
    [amat_annual,pprior,pempirical,Ed,n0,n1,name_features,idx_features,...
        ind_nan_mths,groups] = prepare_wildfire_data(use_quadratic_features);
    m = length(Ed);     % Number of features
end


%% Construct regularization path for the elastic net method
% Sequence of hyperparameters, starting from smallest value /w dual = 0
% If alpha < 0.05, constaint the initial parameter.
if(alpha >= 0.05)
    lambda_est = norm(Ed - amat_annual'*pprior,inf)/alpha;
else
    lambda_est = norm(Ed - amat_annual'*pprior,inf)/0.05;
end
lambda = lambda_est*reg_path;
l_max = length(lambda);


%% Placeholder solutions and quantities for timings
sol_w = zeros(m,l_max);
sol_p = zeros(n0+n1,l_max); sol_p(:,1) = pprior;


%% FIT AN ELASTIC NET REGULARIZED MAXENT MODEL VIA THE FISTA ALGORITHM
if(use_fista)
    disp(' ')
    disp('The FISTA method \w variable selection for elastic net Maxent')
    
    time_total = 0;
    time_iter = 0; 

    % Initialize parameters
    L22 = svds(double(amat_annual),1);
    
    for i=2:1:l_max
        tic
        % Variable selection for elastic net Maxent
        ind = screening_en(lambda(i),lambda(i-1),...
            sol_p(:,i-1),pempirical,alpha,Ed,amat_annual,m);
    
        % Display percentage of zero coefficients
        disp(['Iteration ',num2str(i),'/',num2str(l_max)])
        if(display_output)
            disp(['Percentage of coefficients found to be zero: ',...
                num2str(100-100*sum(ind)/m)])
        end

        %L22 = svds(double(amat_annual(:,ind)),1);

        % Call the FISTA solver
        tau = 1/L22;
        mu = (1-alpha)*(lambda(i));
        q = tau*mu/(1+tau*mu);

        [sol_w(ind,i),sol_p(:,i),num_iter_tot_reg] = ... 
            fista_solver_en(sol_w(ind,i-1),pprior,...
            lambda(i),alpha,amat_annual(:,ind),tau,mu,q,Ed(ind),...
            max_iters,tol);   

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


%% FIT AN ELASTIC NET REGULARIZED MAXENT MODEL VIA COORDINATE DESCENT
if(use_cdescent)
    disp(' ')
    disp('Algorithm: Coordinate Descent (Cortes et al. (2015)) with variable selection')

    time_total = 0;
    time_iter = 0; 

    for i=2:1:l_max
        tic
        % Variable selection for elastic net Maxent
        ind = screening_en(lambda(i),lambda(i-1),...
            sol_p(:,i-1),pempirical,alpha,Ed,amat_annual,m);
    
        % Display percentage of zero coefficients
        disp(['Iteration ',num2str(i),'/',num2str(l_max)])
        if(display_output)
            disp(['Percentage of coefficients found to be zero: ',...
                num2str(100-100*sum(ind)/m)])
        end
    
        % Call the solver for this problem and compute the resultant
        % probability distribution.    
        [sol_w(ind,i),sol_p(:,i),num_iter_tot_reg] = ... 
            cdescent_solver_en(sol_w(ind,i-1),sol_p(:,i-1),lambda(i),...
            alpha,amat_annual(:,ind),Ed(ind),max_iters,tol);   
        time_iter = toc;
        time_total = time_total + time_iter;
        
        % If enabled, the code below will print info about the iteration
        if(display_output)
            disp(['Solution computed for lambda = ', num2str(lambda(i),'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
            disp(['Total time elapsed = ',num2str(time_iter),' seconds.'])
            disp(' ')
        end
    end
    disp('----------')
    disp(['Total time elapsed for constructing the regularization path with the coordinate descent algorithm = ',num2str(time_total),' seconds.'])
    disp('----------')
end


%% FIT AN ELASTIC NET REGULARIZED MAXENT MODEL VIA NPDHG
if(use_npdhg)
    disp(' ')
    disp('The nPGHG method \w variable selection for elastic net Maxent')

    time_total = 0;
    time_iter = 0; 

    for i=2:1:l_max
        tic
        % Variable selection for elastic net Maxent
        ind = screening_en(lambda(i),lambda(i-1),...
            sol_p(:,i-1),pempirical,alpha,Ed,amat_annual,m);
    
        % Display percentage of zero coefficients
        disp(['Iteration ',num2str(i),'/',num2str(l_max)])
        if(display_output)
            disp(['Percentage of coefficients found to be zero: ',...
                num2str(100-100*sum(ind)/m)])
        end

        % Initialize parameters
        L12_sq = max(sum((amat_annual(:,ind).').^2));
        
        % Call the nPDHG solver
        if(alpha > threshold)   % Algorithm sublinear convergence
            theta = 0; tau = 2; sigma = 0.5/L12_sq;

            [sol_w(ind,i),sol_p(:,i),num_iter_tot_reg] = ... 
                npdhg_solver_en_sublinear(sol_w(ind,i-1),pprior,...
                lambda(i),alpha,amat_annual(:,ind),tau,sigma,theta,Ed(ind),...
                max_iters,tol);   
        else                    % Algorithm with linear convergence
            mu = (1-alpha)*lambda(i)*0.5/L12_sq;
            theta = 1-mu*(sqrt(1+2/mu) - 1);
            tau = (1-theta)/theta; 
            sigma = (1-theta)/(theta*(1-alpha)*lambda(i));

            [sol_w(ind,i),sol_p(:,i),num_iter_tot_reg] = ... 
                npdhg_solver_en_linear(sol_w(ind,i-1),pprior,...
                lambda(i),alpha,amat_annual(:,ind),tau,sigma,theta,Ed(ind),...
                max_iters,tol);   
        end
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
    save(strjoin(["data/generated_data/reg_path_alpha=",num2str(alpha),...
        ",min_path=",num2str(reg_path(end)),...
        ",quad_features=",num2str(use_quadratic_features),'.mat'],''),'lambda')
    
    save(strjoin(["data/generated_data/p_sol_alpha=",num2str(alpha),...
        ",min_path=",num2str(reg_path(end)),...
        ",quad_features=",num2str(use_quadratic_features),'.mat'],''),'sol_p')
    
    save(strjoin(["data/generated_data/w_sol_alpha=",num2str(alpha),...
        ",min_path=",num2str(reg_path(end))...
        ",quad_features=",num2str(use_quadratic_features),'.mat'],''),'sol_w')
    % save("data/name_features",'name_features')
    % save("data/groups",'groups')
end


%% Postprocessing
% Identify group thresholds
display_features_results(sol_w,lambda,name_features,groups)

% Plot the regularization path
print_regularization_path(sol_w,lambda,groups);


%% TODO
% Generate lots of fire probabilities plot

% Requirements: the prior distribution; probability solutions; lambda;
% alpha, pempirical; 

% Have a prior plot; elastic net with different alpha values;

% Compare prior vs solutions
% Prior figure; fix alpha and show different hyperparameters; multiple figs
% like that

% Feature importance
%
% Requirements: the solution w; lambda; alpha; groupings (for the group
% lasso), name of the found features;
%
% Maybe send an example of the feature importance?


%% END