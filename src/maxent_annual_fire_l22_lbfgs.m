%% Script written by Gabriel Provencher Langlois
% This script performs l22-regularized Maxent on the fire data set
% provided by JB using L-BFGS as implemented in MATLAB.

% The wildfire data set contains spatial and temporal data about the
% frequency of wildfires in the Western US continent. Here, we are averaging
% the temporal data individually over each month (i.e., spatial data set
% over the months of January through December).


%% Input
% Regularization path. Entries are numbers that multiply a parameter
% estimated from the data.
reg_path = [100,20,10,0.75,0.5,0.25,0.15,0.10,0.05,0.01,0.0075,0.005,0.0025,0.001,0.00075,0.0005,0.00025,0.0001];

% Tolerance for the optimality condition.
tol = 1e-04;           


%% Data extraction
[amat_annual,pprior,pempirical,Ed,n0,n1,~,~] = process_augment_wildfire_data;
amat_annual = double(amat_annual);
pprior = double(pprior);
pempirical = double(pempirical);
Ed = double(Ed);


%% Compute hyperparameters 
% Estimate of lambda parameter to use
val = Ed - amat_annual.'*pprior;
lambda_est = norm(val);

% Compute hyperparameters to be used
lambda = lambda_est*reg_path;


%% Parameters of the algorithm
% Compute the matrix norm for the nPDHG algorithm
tic
L12_sq = max(sum((amat_annual.').^2));
time_L12 = toc;

% Dimensions
m = length(Ed); % Number of features
l_max = length(lambda); % Length of the regularization path

% Placeholders for solutions
% Note: The first columns of these quantities are the initial values
% for the Maxent problem at lambda(1). The solutions are stored
% in the columns 2:l_max+1.

% Placeholders for solutions
% Note: The first columns of these quantities are the initial values
% for the Maxent problem at lambda(1). The solutions are stored
% in the columns 2:l_max+1.
sol_npdhg_w = zeros(m,l_max+1); sol_npdhg_w(:,1) = val./lambda(1);

% Timings and Maximum number of iterations
max_iter = 2000;
time_lbfgs = 0;
time_lbfgs_total = 0;


%% Script for the lBFGS algorithm
disp(' ')
disp('Algorithm: The lBFGS method (with regular sequence)')

% Regularization path
for i=1:1:l_max
    tic
    
    % Call the solver for this problem
    t = lambda(i);
    sol_npdhg_w(:,i+1) = lbfgs_solver(sol_npdhg_w(:,i),t,amat_annual,Ed);
    time_lbfgs = toc;
    Ed_minus_Ep = Ed - (compute_p(amat_annual,sol_npdhg_w(:,i+1)).'*amat_annual).';
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '.'])
    disp(['Total time elapsed = ',num2str(time_lbfgs),' seconds.'])
    disp(['Relative l22 deviation from presence-only data: ', num2str(norm(Ed_minus_Ep)/norm(Ed))])
    disp(' ')
    time_lbfgs_total = time_lbfgs_total + time_lbfgs;
end
disp(['Total time elapsed for the l-BFGS method = ',num2str(time_lbfgs_total + time_lbfgs),' seconds.'])
% End of the script.



%% Auxiliary functions
function p = compute_p(A,w)
% Compute a probability vector p from the formula
% p(j) = exp([A*w]_{j})/(sum_{j}exp([A*w]_{j}))
% for every j in {1,...,n} and vector w.
% The matrix A is an m x n matrix.

x = A*w;
w = exp(x-max(w));
p = w/sum(w);
end



%% Solver
function sol_w = lbfgs_solver(w0,lambda,A,Ed)
% Nonlinear PDHG method for solving Maxent with 0.5*normsq{\cdot}.
% Input variables:
%   w = Array of dimension m x 1. This is the starting point.
%   lambda = Positive scalar.
%   A = An m x n matrix.
%   Ed = Observed features of presence-only data.

options = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'MaxIterations',400,'OptimalityTolerance',1e-04,'Display','off');
sol_w = fminunc(@l2maxent,w0,options);

    function [f,g] = l2maxent(w)
        % Note: The prior is assumed to be uniform.
        % Compute the objective function
        p = A*w; a = max(p);
        p = exp(p-a); b = sum(p);
        log_term = a + log(b);
        f= log_term + (0.5*lambda)*sum((w.^2)) - Ed'*w;
       
        % Compute the gradient of the objective function
        g = (p.'*A).'/b;
        g = g + lambda*w - Ed;
    end
end