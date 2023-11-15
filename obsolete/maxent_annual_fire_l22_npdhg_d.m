%% Script written by Gabriel Provencher Langlois
% This script performs l22-regularized Maxent on the fire data set
% provided by JB using the nPDHG algorithm developed by GPL and JD.

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

% Compute the matrix norm for the nPDHG algorithm
tic
L12_sq = max(sum((amat_annual.').^2));
time_L12 = toc;


%% Parameters of the algorithm
% Dimensions
m = length(Ed); % Number of features
l_max = length(lambda); % Length of the regularization path

% Strong convexity factor
gamma_g = 1;

% Placeholders for solutions
% Note: The first columns of these quantities are the initial values
% for the Maxent problem at lambda(1). The solutions are stored
% in the columns 2:l_max+1.
sol_npdhg_w = zeros(m,l_max+1); sol_npdhg_w(:,1) = val./lambda(1);
sol_npdhg_z = zeros(m,l_max+1);
sol_npdhg_p = zeros(n0+n1,l_max+1); sol_npdhg_p(:,1) = pprior;

% Timings and Maximum number of iterations
time_npdhg_regular = 0;
time_npdhg_total = 0;
max_iter = 1000;


%% Script for the nPDHG algorithm
disp(' ')
disp('Algorithm: The nPGHG method (with regular sequence)')

% Regularization path
for i=1:1:l_max
    tic
    
    % Initialize the regularization hyperparameter and other parameters
    t = lambda(i); 
    gamma_h = t;
    mu = 0.5*gamma_g*gamma_h/L12_sq;
    theta = 1 - mu*(sqrt(1 + 2/mu)-1);
    tau = (1-theta)/(gamma_g*theta);
    sigma = 1/(theta*tau*L12_sq);

    % Call the solver for this problem and compute the resultant
    % probability distribution
    [sol_npdhg_w(:,i+1),sol_npdhg_z(:,i+1),sol_npdhg_p(:,i+1),Ed_minus_Ep,num_iter_tot_reg] = ... 
        solver_l22_npdhg(sol_npdhg_w(:,i),sol_npdhg_z(:,i),t,amat_annual,tau,sigma,theta,Ed,max_iter,tol);   
    time_npdhg_regular = toc;
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_npdhg_regular),' seconds.'])
    disp(['Relative l22 deviation from presence-only data: ', num2str(norm(Ed_minus_Ep)/norm(Ed))])
    disp(' ')
    time_npdhg_total = time_npdhg_total + time_npdhg_regular;
end
disp(['Total time elapsed for the nPDHG method = ',num2str(time_npdhg_total + time_L12),' seconds.'])
% End of the script.