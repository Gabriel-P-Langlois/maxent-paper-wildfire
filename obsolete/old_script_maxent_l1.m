%% Script written by Gabriel Provencher Langlois
% This script performs l1-regularized Maxent on the fire data set
% provided by Jatan Buch using the either the nPDHG algorithm developed by
% Gabriel P. Langlois or the coordinate descent method of Cortes and al.

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. 

% Note: We are averaging the temporal data individually over each month 
% (i.e., spatial data set over the months of January through December).


%% Notes


%% Options for the script
% Option to use nPDHG vs. coordinate descent algorithm
use_npdhg = true;

% Option to use quadratic features
use_quadratic_features = false;

% Tolerance for the optimality condition.
tol = 1e-04;             


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
    [amat_annual,pprior,pempirical,Ed,n0,n1,name_features,idx_features,ind_nan_mths] = prepare_wildfire_data(use_quadratic_features);
end


%% Initiate regularization path (hyperparameters)
% Initiate the regularization path as an array.
% The entries must be positive and decreasing numbers starting from one.
reg_path = [1:-0.01:0.75,0.745:-0.005:0.60];

%reg_path = [1:-0.01:0.75,0.745:-0.005:0.50, ...
%     0.4975:-0.0025:0.35];%,0.349:-0.001:0.20];

% reg_path = [1:-0.01:0.75,0.745:-0.005:0.50,...
%      0.4925:-0.0025:0.35,0.349:-0.001:0.20];

% reg_path = [1,0.96,0.95:-0.01:0.75,0.745:-0.005:0.50,...
%      0.4925:-0.0025:0.35,0.349:-0.001:0.20,0.19995:-0.0005:0.15];

% reg_path = [1,0.96,0.95:-0.01:0.75,0.745:-0.005:0.50,...
%     0.4925:-0.0025:0.35,0.349:-0.001:0.20,0.19995:-0.0005:0.125,...
%     0.12475:-0.00025:0.05];

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
time_iter = 0;
time_total = 0;

% Other quantities for the nPDHG algorithm
theta = 0; tau = 2;


%% Solve l1-regularized Maxent using nPDHG algorithm or coordinate descent
if(use_npdhg)
    disp(' ')
    disp('Algorithm: The nPGHG method with variable selection')
    max_iter = 4000;
    
    % Regularization path
    time_L12 = 0;
    for i=2:1:l_max
        tic
        disp(['Iteration ',num2str(i),'/',num2str(l_max)])
        t = lambda(i); 
        
        % Variable selection for l1-Maxent (based on unpublished results)
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
        time_iter = toc;
        
        % Display outcome
        disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of primal-dual steps = ', num2str(num_iter_tot_reg), '.'])
        disp(['Total time elapsed = ',num2str(time_iter),' seconds.'])
        disp(' ')
        time_total = time_total + time_iter;
    end
    disp(['Total time elapsed for the nPDHG method = ',num2str(time_total + time_L12),' seconds.'])
else
    disp(' ')
    disp('Algorithm: Coordinate Descent (Cortes et al. (2015) with variable selection')
    max_iter = 20000;

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
    
        ind = (lhs >= rhs);
        
        % Display coefficients found to be zero.
        disp(['Percentage of coefficients found to be zero: ',num2str(100-100*sum(ind)/m)])
    
        % Call the solver for this problem and compute the resultant
        % probability distribution.    
        [sol_npdhg_w(ind,i),sol_npdhg_p(:,i),num_iter_tot_reg] = ... 
            solver_l1_cd(sol_npdhg_w(ind,i-1),s,t,amat_annual(:,ind),Ed(ind),max_iter,tol);   
        time_iter = toc;
        
        % Display outcome
        disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
        disp(['Total time elapsed = ',num2str(time_iter),' seconds.'])
        disp(' ')
        time_total = time_total + time_iter;
    end
    disp(['Total time elapsed for the CD method = ',num2str(time_total),' seconds.'])
end


%% Postprocessing
idx_lambda_found = postprocess_wildfire_data(sol_npdhg_w,reg_path,name_features);


%% (WIP) Computing the probability map
% Convert the probability vector into an h5 file that Jatan can process
% using his Python code. Jatan has the function for that.
%: Must be of double type!

% Note: Data must be saved as double.

% Repopulate the probability vector with its NaN values
% ind_nan_mths = reshape(ind_nan_mths,[length(ind_nan_mths)*12,1]);
% prob_vector_to_save = zeros(length(ind_nan_mths),1);
% prob_vector_to_save(ind_nan_mths) = NaN;
% prob_vector_to_save(~ind_nan_mths) = sol_npdhg_p(:,end);
% 
% h5create('my_example_file2.h5', '/data', size(prob_vector_to_save));
% h5write('my_example_file2.h5', '/data', double(prob_vector_to_save));
% test = h5read('my_example_file2.h5', '/data');

% Visualize the probability vector


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% END OF THE SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Helper function II: Solver for the l1-coordinate descent algorithm
function [sol_w,sol_p,k] = solver_l1_cd(w,p0,lambda,A,Ed,max_iter,tol)
% Nonlinear PDHG method for solving Maxent with lambda*norm(\cdot)_{1}
% Input variables:
%   w: m x 1 vector -- Weights of the gibbs distribution.
%   p: n x 1 vector -- Parameterization of the gibbs probability
%   distribution, where p(j) = pprior(j)e^{u(j)-C}.
%   lambda: Positive number -- Hyperparameter.
%   A: n x m matrix -- Matrix of features (m) for each grid point (n).
%   tau, sigma, theta: Positive numbers -- Stepsize parameters.
%   Ed: m-dimensional vector -- Observed features of presence-only data. 
%   max_iter: Positive integer -- Maximum number of iterations.
%   tol:    Small number -- used for the convergence criterion

% Output:
%   sol_w: m x 1 column vector -- dual solution
%   sol_p: n x 1 column vector -- primal solution


% Auxiliary variables -- For the algorithm
tmp1 = ((p0'*A)') - Ed;
tmp2 = A*w;
wplus = w;

% Main algorithm
k = 0; 
flag_convergence = true;
d = zeros(length(w),1);

while (flag_convergence)
    % Update counter
    k = k + 1;
    
    % Apply thresholding
    for i=1:1:length(w)
       if(w(i)~=0)
           d(i) = lambda*sign(w(i)) + tmp1(i);
       elseif(abs(tmp1(i)) < lambda)
           d(i) = 0;
       else
           d(i) = -lambda*sign(w(i)) + tmp1(i);
       end
    end
    [~,j_ind] = max(abs(d));
    
    % Approximate steepest descent
    if(abs(w(j_ind)-tmp1(j_ind)) < lambda)
        eta = -w(j_ind);
    elseif(w(j_ind)-tmp1(j_ind) > lambda)
        eta = -(lambda+tmp1(j_ind));
    else
        eta = (lambda-tmp1(j_ind));
    end
    
    % Update dual variable + scalar product
    wplus(j_ind) = w(j_ind) + eta;
    tmp2 = tmp2 + eta*A(:,j_ind);

    % Update the probability and the difference in expectations
    pplus = exp(tmp2-max(tmp2)); norm_sum = sum(pplus);
    tmp1 = ((pplus'*A)')/norm_sum - Ed;
    
    % Increment dual variable
    w = wplus;

    % Convergence check -- We use the optimality condition on the l1 normd
    flag_convergence = ~(((k >= 40) && (norm(tmp1,inf) <= lambda*(1 + tol))) || (k >= max_iter));
end

% Final solutions
sol_p = pplus/norm_sum;
sol_w = wplus;
end