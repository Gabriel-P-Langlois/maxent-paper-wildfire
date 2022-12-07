%% Script written by Gabriel Provencher Langlois
% This script performs l22-regularized Maxent on the fire data set
% provided by JB using gradient descent with LBGFS.

% TODO: Add details on the LBFGS method.
    % - Continue implementation
    % - Implement GD version + Wolfe line search. See if it converges. If
    % it does, include the line search as part of a mini function.



%% Input
% Regularization path. Entries are numbers that multiply lambda_est.
% This script solves l22-regularized Maxent with the hyperparameters
% reg_path*lambda in an efficient manner.
reg_path = [1000,100,50,25]; 

% Number of vectors approximating the hessian of the primal problem
num_lbfgs = 10; 


%% Extract the data and prepare it
% block0_values: 38 Features (see bottom of the script)
% block1_values: 7 Features (see bottom of the script)
% block2_values: 5 quantities (see bottom of the script)

% The features and quantities are listed at the bottom of the script,
% after the code for the solver and before the misc information.

% Read the data
% Note: The design matrix is an (n x m) matrix.
A = [h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block0_values')',single(h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block1_values'))'];
data8 = h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block2_values')';

% Remove NaN values
ind_nan = isnan(A(:,1));
A = A(~ind_nan,:);
data8 = data8(~ind_nan,:);

% Separate data with fires and those with no fire.
ind_fire_yes = (data8(:,1) >= 1);

% Compute the number of background and presence-only data points
n0 = length(unique(data8(~ind_fire_yes,[2,4:5]),'rows'));
n1 = length(unique(data8(ind_fire_yes,[2,4:5]),'rows'));

% Normalize the features.
% Note: Background features are normalized together.
means_A = mean(A,1); A = A - means_A;
sumsq_A = sum(A.^2); A = A./sqrt(sumsq_A/(n0+n1)); % Normalization
b = A(ind_fire_yes,:);
A = A(~ind_fire_yes,:);

% Average over the features of the presence only data. We take the average
% w.r.t. the uniform distribution with n1 elements.
Ed = b'*ones(n1,1)/n1;

% Compute an estimate of the parameter (for scaling purposes).
% The prior distribution is uniform w.r.t. all background samples, i.e.,
% ones(n0,1)/n0;
lambda_est = norm(Ed - A'*(ones(n0,1)/n0));

% Compute hyperparameters to be used
lambda = lambda_est*reg_path;

% Clear irrelevant arrays
clear data8 ind_fire_yes ind_nan



%% Parameters of the algorithm
% Dimensions
m = length(Ed); % Number of features
n = n0;         % Number of background samples
l_max = length(lambda); % Length of the regularization path

% Placeholders for solutions
% Note: The first columns of these quantities are the initial values
% for the Maxent problem at lambda(1). The solutions are stored
% in the columns 2:l_max+1.

% Note: L-BFGS does not make use of the structure of the problem.
sol_npdhg_w = zeros(m,l_max+1);
sol_npdhg_p = zeros(n,l_max+1); sol_npdhg_p(:,1) = compute_p(A,sol_npdhg_z(:,1));

% Timings and Maximum number of iterations
time_lbfgs_regular = 0;
time_lbfgs_total = 0;
max_iter = 2000;



%% Script for the lBFGS algorithm
disp(' ')
disp('Algorithm: The lBFGS method (with regular sequence)')

% Regularization path
for i=1:1:l_max
    tic
    
    % Call the solver for this problem
    t = lambda(i);
    [sol_npdhg_w(:,i+1),num_iter_tot_reg] = lbfgs_solver(sol_npdhg_w(:,i),t,A,Ed,num_lbfgs,max_iter);
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_lbfgs_regular),' seconds.'])
    disp(['Relative l22 deviation from presence-only data: ',num2str(norm(Ed - A'*sol_npdhg_p(:,i+1))/norm(Ed))]) % Measure of how regularized the problem is.
    disp(' ')
    time_lbfgs_total = time_lbfgs_total + time_lbfgs_regular;
end
disp(['Total time elapsed for the l-BFGS method = ',num2str(time_lbfgs_total + time_L12),' seconds.'])
%% End of the script.



%% Auxiliary functions
function p = compute_p(A,w)
% Compute a probability vector p from the formula
% p(j) = exp([A*w]_{j})/(sum_{j}exp([A*w]_{j}))
% for every j in {1,...,n} and vector w.

x = A*w; a = max(x);
w = exp(x-a);
p = w/sum(w);
end



%% Solver
function [sol_w,num_iter_tot] = lbfgs_solver(w,lambda,A,Ed,num_lbfgs,max_iter)
% Nonlinear PDHG method for solving Maxent with 0.5*normsq{\cdot}.
% Input variables:
%   w = Array of dimension m x 1. This is the starting point.
%   lambda = Positive scalar.
%   A = An n x m matrix.
%   Ed = Observed features of presence-only data.
%   num_lbfgs = Number of points for the L-BFGS method.
%   max_iter = Integer that puts a cap on the number of iterations.

% Auxiliary variables
s = zeros(m,num_lbfgs); % Stores solution iterates
y = zeros(m,num_lbfgs); % Stores gradient iterates
num_iter_tot = 0;       % Counter for the iterations

% First iteration with plain gradient descent
    grad_k = grad_primal(w,lambda,A,Ed);
    tau = 1/sqrt(max(sum((A').^2))); % Descent step -- CHECK
    
    % Perform a single step of gradient descent and save results
    wplus = w  - tau*grad_k;
    grad_kplusone = grad_primal(wplus,lambda,A,Ed);
    
    % Indices are 0:1:19. Annoying, but unfortunately necessary.
    s(:,1) = wplus - w;
    y(:,1) = grad_kplusone - grad_k;

% Iterations after the first one.
for k=1:1:max_iter-1
    num_iter_tot = num_iter_tot + 1;
    
    % Choose Hessian matrix
    %gamma = ()/norm(); % USe mod function
    Ik = gamma*eye(m,m);
end

% Final solution
sol_w = w;



    function grad = grad_primal(w,lambda,A,Ed)
        % Compute gradient of the log term
        x = A*w; a = max(x);
        p = exp(x-a);
        p = p/sum(p);
        
        % Compute gradient of the primal problem
        grad = A'*p;
        grad = grad + lambda*w - Ed;
    end

    function r = lbfgs_step(k,num_lbfgs,gamma,q,s,y)
        % k: Current iterate
        % num_lbfgs: max sixe of approximate hessian
        % gamma = estimate of the size of the 
        % q = \nabla f(w_{k})
        % s = m x num_lbfgs matrix
        % y = m x num_lbfgs matrix
        
        % First for loop
        i = k-1;
        alpha = zeros(i,1);
        while(i >= (k-num_lbfgs))
            % COMPLETE.
            i = i-1; 
        end
        
        
        r = 0;
    end
end


%% Information regarding the wildfire data set.
% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block0_items')
% 
% ans =
% 
%   38×1 cell array
%     {'Tmax       '}
%     {'VPD        '}
%     {'Prec       '}
%     {'Solar      '}
%     {'Wind       '}
%     {'Elev       '}
%     {'RH         '}
%     {'FM1000     '}
%     {'Ant_Tmax   '}
%     {'AvgVPD_3mo '}
%     {'Avgprec_3mo'}
%     {'Ant_RH     '}
%     {'CAPE       '}
%     {'FFWI_max3  '}
%     {'FFWI_max7  '}
%     {'Tmin       '}
%     {'Camp_dist  '}
%     {'Camp_num   '}
%     {'Road_dist  '}
%     {'Avgprec_4mo'}
%     {'Avgprec_2mo'}
%     {'VPD_max3   '}
%     {'VPD_max7   '}
%     {'Tmax_max3  '}
%     {'Tmax_max7  '}
%     {'Tmin_max3  '}
%     {'Tmin_max7  '}
%     {'Slope      '}
%     {'Southness  '}
%     {'AvgVPD_4mo '}
%     {'AvgVPD_2mo '}
%     {'SWE_mean   '}
%     {'SWE_max    '}
%     {'AvgSWE_3mo '}
%     {'Delta_T    '}
%     {'Biomass    '}
%     {'Lightning  '}
%     {'RH_min3    '}

% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block1_items')
% 
% ans =
% 
%   7×1 cell array
% 
%     {'Antprec_lag1'}
%     {'Forest      '}
%     {'Grassland   '}
%     {'Urban       '}
%     {'Antprec_lag2'}
%     {'Popdensity  '}
%     {'Housedensity'}

% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block2_items')
% 
% ans =
% 
%   5×1 cell array
% 
%     {'fire_freq'}
%     {'month    '}
%     {'reg_indx '}
%     {'X        '}
%     {'Y        '}