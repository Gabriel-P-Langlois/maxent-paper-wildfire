%% Script written by Gabriel Provencher Langlois
% This script fits an elastic net regularized Maxent model on the fire data
% set provided by Jatan Buch using the either the nPDHG algorithm developed
% by Gabriel P. Langlois, the FISTA algorithm (Beck and Teboulle, 2009)
% or the coordinate descent method of Cortes and al. (2015)

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. 


%% Notes
% 1) Run the script from the project directory where ./data is located.
% 2) (Minor issue) The first step of the npdhg algorithm is sometimes slow
%    This seems to occurs when specifying many points early along
%    the regularization path.


%% Options for the script
% Only modify the values below and not anywhere else in the script

% Check for a new run. Set to 1 if you are doing a new run, 0 otherwise.
% If set to 0, the script will not attempt to load the data
new_run = 1;

% Option to output the result at each iteration if desired
display_output = true;

% Option to save the results at the end if desired
save_results = false;

% Specify which algorithm to use. Note: Only one should be specified.
use_fista = false;
use_cdescent = false;
use_npdhg = true;

% Option to use quadratic features
use_quadratic_features = false;

% Elastic net parameter and initialize structure of the regularization path
% For testing, use alpha = 1.0, npts_path = 100;, min_val_path = 0.70;
alpha = 1.0;
npts_path = 200;
min_val_path = 0.50;

reg_path = linspace(1,min_val_path,npts_path);

% Threshold for using linear vs sublinear (sublinear if alpha > threshold)
threshold = 0.30;

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
sol_w = single(zeros(m,l_max));
sol_p = single(zeros(n0+n1,l_max)); sol_p(:,1) = pprior;


%% FIT AN ELASTIC NET REGULARIZED MAXENT MODEL VIA THE FISTA ALGORITHM
% TODO: Estimation of the L22 norm...
if(use_fista)
    disp(' ')
    disp('The FISTA method \w variable selection for elastic net Maxent')
    
    time_total = 0;
    time_iter = 0; 
    
    for i=2:1:l_max
        tic
        % Variable selection for elastic net Maxent
        [ind,~] = screening_en(lambda(i),lambda(i-1),...
            sol_p(:,i-1),pempirical,alpha,Ed,amat_annual,m);
    
        % Display percentage of zero coefficients
        if(display_output)
            disp(['Iteration ',num2str(i),'/',num2str(l_max)])
            disp(['Percentage of coefficients found to be zero: ',...
                num2str(100-100*sum(ind)/m)])
        end

        % Initialize parameters
        L22 = svds(double(amat_annual(:,ind)),1);
        
        % Call the FISTA solver
        tau = 1/L22;
        mu = (1-alpha)*(lambda(i));
        q = tau*mu/(1+tau*mu);

        [sol_w(ind,i),sol_p(:,i),num_iter_tot_reg] = ... 
            fista_solver_en(sol_w(ind,i-1),pprior,...
            lambda(i),alpha,amat_annual(:,ind),tau,mu,q,Ed(ind),...
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


%% FIT AN ELASTIC NET REGULARIZED MAXENT MODEL VIA COORDINATE DESCENT
if(use_cdescent)
    disp(' ')
    disp('Algorithm: Coordinate Descent (Cortes et al. (2015)) with variable selection')

    time_total = 0;
    time_iter = 0; 

    for i=2:1:l_max
        tic
        % Variable selection for elastic net Maxent
        [ind,p_conv] = screening_en(lambda(i),lambda(i-1),...
            sol_p(:,i-1),pempirical,alpha,Ed,amat_annual,m);
    
        % Display percentage of zero coefficients
        if(display_output)
            disp(['Iteration ',num2str(i),'/',num2str(l_max)])
            disp(['Percentage of coefficients found to be zero: ',...
                num2str(100-100*sum(ind)/m)])
        end
    
        % Call the solver for this problem and compute the resultant
        % probability distribution.    
        [sol_w(ind,i),sol_p(:,i),num_iter_tot_reg] = ... 
            cdescent_solver_en(sol_w(ind,i-1),p_conv,lambda(i),...
            alpha,amat_annual(:,ind),Ed(ind),max_iter,tol);   
        time_iter = toc;
        time_total = time_total + time_iter;
        
        if(display_output) % Display outcome
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
        [ind,p_conv] = screening_en(lambda(i),lambda(i-1),...
            sol_p(:,i-1),pempirical,alpha,Ed,amat_annual,m);
    
        % Display percentage of zero coefficients
        if(display_output)
            disp(['Iteration ',num2str(i),'/',num2str(l_max)])
            disp(['Percentage of coefficients found to be zero: ',...
                num2str(100-100*sum(ind)/m)])
        end

        % Initialize parameters
        L12_sq = max(sum((amat_annual(:,ind).').^2));
        
        % Call the nPDHG solver
        if(alpha > threshold)    % Algorithm sublinear convergence
            theta = 0; tau = 2; sigma = 0.5/L12_sq;
            u_in = log(p_conv./pprior);

            [sol_w(ind,i),sol_p(:,i),num_iter_tot_reg] = ... 
                npdhg_solver_en_sublinear(sol_w(ind,i-1),u_in,...
                lambda(i),alpha,amat_annual(:,ind),tau,sigma,theta,Ed(ind),...
                max_iter,tol);   
        else                        % Algorithm with linear convergence
            mu = (1-alpha)*lambda(i)*0.5/L12_sq;
            theta = 1-mu*(sqrt(1+2/mu) - 1);
            tau = (1-theta)/theta; 
            sigma = (1-theta)/(theta*(1-alpha)*lambda(i));

            [sol_w(ind,i),sol_p(:,i),num_iter_tot_reg] = ... 
                npdhg_solver_en_linear(sol_w(ind,i-1),log(p_conv./pprior),...
                lambda(i),alpha,amat_annual(:,ind),tau,sigma,theta,Ed(ind),...
                max_iter,tol);   
        end
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
    save(strjoin(["data/generated_data/reg_path_alpha=",num2str(alpha),",npts=",...
        num2str(npts_path),",min_path=",num2str(min_val_path),...
        ",quad_features=",num2str(use_quadratic_features),'.mat'],''),'lambda')
    
    save(strjoin(["data/generated_data/p_sol_alpha=",num2str(alpha),",npts=",...
        num2str(npts_path),",min_path=",num2str(min_val_path),...
        ",quad_features=",num2str(use_quadratic_features),'.mat'],''),'sol_p')
    
    save(strjoin(["data/generated_data/w_sol_alpha=",num2str(alpha),",npts=",...
        num2str(npts_path),",min_path=",num2str(min_val_path)...
        ",quad_features=",num2str(use_quadratic_features),'.mat'],''),'sol_w')
    % save(strjoin["data/generated_data/name_features"],'name_features')
end



%% END
%% TODO
% Generate lots of fire probabilities plot
%
% Requirements: the prior distribution; probability solutions; lambda;
% alpha, pempirical; 
%
%
% Have a prior plot; elastic net with different alpha values; idem for
% NOGL and linf
% Compare prior vs solutions
% Prior figure; fix alpha and show different hyperparameters; multiple figs
% like that

% Feature importance
%
% Requirements: the solution w; lambda; alpha; groupings (for the group
% lasso), name of the found features;
%
% Maybe send an example of the feature importance?