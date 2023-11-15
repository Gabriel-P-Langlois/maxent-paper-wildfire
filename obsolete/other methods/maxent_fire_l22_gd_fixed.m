%% Script written by Gabriel Provencher Langlois
% This script performs l22-regularized Maxent on the fire data set
% provided by JB using gradient descent with fixed step.



%% Input
% Regularization path. Entries are numbers that multiply lambda_est.
reg_path = [1000,100,50,25,10]; 



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
sol_gd_w = zeros(m,l_max+1);
sol_gd_p = zeros(n,l_max+1); sol_gd_p(:,1) = compute_p(A,sol_gd_w(:,1));

% Timings and Maximum number of iterations
time_gd_regular = 0;
time_gd_total = 0;
max_iter = 2000;



%% Script
disp(' ')
disp('Algorithm: Gradient descent with fixed step size')
    
% Compute initial stepsize
L12_sq = max(sum((A').^2));

% Regularization path
for i=1:1:l_max
    tic
    
    % Initialize the regularization hyperparameter and other parameters
    t = lambda(i); 

    % Call the solver for this problem
    [sol_gd_w(:,i+1),num_iter_tot_reg] = gd_solver(sol_gd_w(:,i),t,A,(1/L12_sq),Ed,max_iter);   
    sol_gd_p(:,i+1) = compute_p(A,sol_gd_w(:,i+1));
    time_gd_regular = toc;
    
    % Display outcome
    disp(['Solution computed for lambda = ', num2str(t,'%.4e'), '. Number of iterations = ', num2str(num_iter_tot_reg), '.'])
    disp(['Total time elapsed = ',num2str(time_gd_regular),' seconds.'])
    disp(['Relative l22 deviation from presence-only data: ',num2str(norm(Ed - A'*sol_gd_p(:,i+1))/norm(Ed))]) % Measure of how regularized the problem is.
    disp(' ')
    time_gd_total = time_gd_total + time_gd_regular;
end
disp(['Total time elapsed for the AGD method with fixed stepsize= ',num2str(time_gd_total + time_L12),' seconds.'])
%% End of the script.



%% Auxiliary functions
function p = compute_p(A,w)
% Compute a probability vector p from the formula
% p(j) = exp([A*z]_{j})/(sum_{j}exp([A*z]_{j}))
% for every j in {1,...,n} and some parameter z.

x = A*w; a = max(x);
w = exp(x-a);
p = w/sum(w);
end



%% Solver
function [sol_w,num_iter_tot] = gd_solver(w,lambda,A,tau,Ed,max_iter)
% Input variables:
%   w = Array of dimension m x 1
%   lambda = parameter > 0
%   A = An n x m matrix.
%   tau = initial step size
%   Ed = Observed features of presence-only data. The features are averaged
%   w.r.t. a uniform distribution.

% Auxiliary variables
t = 0;
wminus = w;

% Counter for the iterations
num_iter_tot = 0;

% Iterations
for k=1:1:max_iter
    num_iter_tot = num_iter_tot + 1;
    
    % Extragradient step
    tplus = 0.5*(1+sqrt(1+4*(t^2)));
    y = w +(t/tplus)*(w-wminus);
    
    % AGD with fixed step
    wplus = y - tau*grad_primal(y,lambda,A,Ed);
 
    % Check for convergence
    if((k >= 20) && (mod(k,10) == 0))
        disp(norm(wplus-w)/norm(w))
        if((norm(wplus-w) < (1e-04)*norm(w)))
            break
        end
    end
    
    % Increment
    wminus = w; w = wplus;
end

% Final solutions
sol_w = wplus;

    function grad = grad_primal(w_prob,lambda,A,Ed)
        % Compute gradient of the log term
        x = A*w_prob; a = max(x);
        p = exp(x-a);
        p = p/sum(p);
        
        % Compute gradient of the primal problem
        grad = A'*p;
        grad = grad + lambda*w_prob - Ed;
    end
end