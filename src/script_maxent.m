%% Script written by Gabriel Provencher Langlois
% This script fits an elastic net regularized Maxent model on the fire data
% set provided by Jatan Buch using the either the nPDHG algorithm developed
% by Gabriel P. Langlois or the coordinate descent method of Cortes and al.

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. 

% Note: We are averaging the temporal data individually over each month 
% (i.e., spatial data set over the months of January through December).


%% Notes
% TODO:
%   1) Update the manuscript and the section concerning the elastic net
%   Maxent model. Most of it can be filled now, subject to changes to the
%   methodology.

%   2) Next step is to finish the postprocesseing aspect with the nPDHG
%   algorithm (at least for the elastic net). Finish the analysis that
%   Jatan wants here. We'll do the analysis for alpha = 0.95, 0.5 and maybe
%   0.25.

%   3) While the analysis is underway, GPL can work on the other
%   algorithms.

%   4) Verify that the selection rule for the elastic net is correct.

%% Options for the script
% Elastic net parameter
alpha = 0.95;

% threshold for using linear vs sublinear
alpha_thresh = 0.35;    % use sublinear if alpha > alpha_thresh

% Specify which algorithms to use
use_npdhg = true;
use_fista = false;
use_cdescent = false;

% Option to use quadratic features
use_quadratic_features = false;

% Tolerance for the optimality condition (for all methods)
tol = 1e-4;


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

new_run = 1;
if(new_run)
    % Read the data. Since it's too large, it is stored locally
    % On GPL computer, the data is located at ./.. from the project
    % directory.
    [amat_annual,pprior,pempirical,Ed,n0,n1,name_features,idx_features,...
        ind_nan_mths] = prepare_wildfire_data(use_quadratic_features);
    m = length(Ed); % Number of features
end


%% Construct regularization path for the elastic net method
% Initiate the regularization path.
% The entries must be positive and decreasing numbers starting from one.
reg_path = linspace(1,0.35,100);

% Sequence of hyperparameters, starting from smallest value /w dual = 0
% Must aboid setting alpha close to zero. 
if(alpha >= 0.05)
    lambda_est = norm(Ed - amat_annual'*pprior,inf)/alpha;
else
    lambda_est = norm(Ed - amat_annual'*pprior,inf)/0.05;
end
lambda = lambda_est*reg_path;
l_max = length(lambda);
max_iter = 50000;


%% FIT AN ELASTIC NET REGULARIZED MAXENT MODEL TO THE REG. PATH VIA NPDHG
if(use_npdhg)
    % Initialize placeholders for the solutions and calculating timings
    sol_npdhg_w = single(zeros(m,l_max));
    sol_npdhg_p = single(zeros(n0+n1,l_max)); sol_npdhg_p(:,1) = pprior;

    time_L12 = 0;
    time_iter = 0;
    time_total = 0;

    disp(' ')
    disp('The nPGHG method \w variable selection for elastic net Maxent')
    for i=2:1:l_max
        tic
        disp(['Iteration ',num2str(i),'/',num2str(l_max)])

        % Variable selection for l1-Maxent
        [ind,p_conv] = screening_en(lambda(i),lambda(i-1),...
            sol_npdhg_p(:,i-1),pempirical,alpha,Ed,amat_annual,m);
    
        % Display percentage of zero coefficients
        disp(['Percentage of coefficients found to be zero: ',...
            num2str(100-100*sum(ind)/m)])

        % Initialize parameters
        tic
        L12_sq = max(sum((amat_annual(:,ind).').^2));
        time_L12 = time_L12 + toc;
        
        % Call the nPDHG solver
        if(alpha > alpha_thresh)    % Algorithm sublinear convergence
            theta = 0; tau = 2; sigma = 0.5/L12_sq;

            [sol_npdhg_w(ind,i),sol_npdhg_p(:,i),num_iter_tot_reg] = ... 
                npdhg_solver_en_sublinear(sol_npdhg_w(ind,i-1),log(p_conv./pprior),...
                lambda(i),alpha,amat_annual(:,ind),tau,sigma,theta,Ed(ind),...
                max_iter,tol);   
        else                        % Algorithm with linear convergence
            mu = (1-alpha)*lambda(i)*0.5/L12_sq;
            theta = 1-mu*(sqrt(1+2/mu) - 1);
            tau = (1-theta)/theta; 
            sigma = (1-theta)/(theta*(1-alpha)*lambda(i));

            [sol_npdhg_w(ind,i),sol_npdhg_p(:,i),num_iter_tot_reg] = ... 
                npdhg_solver_en_linear(sol_npdhg_w(ind,i-1),log(p_conv./pprior),...
                lambda(i),alpha,amat_annual(:,ind),tau,sigma,theta,Ed(ind),...
                max_iter,tol);   
        end
        time_iter = toc;

        % Display outcome
        disp(['Solution computed for lambda = ', num2str(lambda(i),'%.4e'), '. Number of primal-dual steps = ', num2str(num_iter_tot_reg), '.'])
        disp(['Time elapsed for solving the Maxent problem = ',...
            num2str(time_iter),' seconds.'])
        disp(' ')
        time_total = time_total + time_iter;
    end
    disp('----------')
    disp(['Total time elasped for constructing the regularization path with the nPDHG algorithm: ',num2str(time_total), ' seconds.'])
    disp('----------')
end


%% END