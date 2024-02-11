%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This short script is used to compute the running time of our algorithm
%   on the wildfire dataset.
%
%   We use the path: 
%       reg_path = [1:-0.01:0.5,...
%       0.495:-0.005:0.10];
%
%   For the elastic net problem, we use alpha = 0.95 and alpha = 0.4.
%
%   For NOGL and INF, we just use the paths as they are.
%
%   Since there are three numerical methods, this gives us 12 separate run.
%
%   Each run is done 10 times. The reported times are calculated as the
%   average over these 10 runs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Notes
% For elastic net with different alphas, try with and without variable
% selection.
% reg_path = [1:-0.01:0.5,...
%       0.495:-0.005:0.10];

% nPDHG / alpha = 0.95 (with):      ~ 300 seconds
% nPDHG / alpha = 0.95 (without):   ~ 316 seconds
% nPDHG / alpha = 0.40 (with):      ~ 119.4031 seconds
% nPDHG / alpha = 0.40 (without):   ~ 126 seconds

% FISTA / alpha = 0.95 (with + mult L22):    3587.8331   ~  seconds
% FISTA / alpha = 0.95 (without):  3429.9091 ~  seconds

% FISTA / alpha = 0.40 (with + mult L22):    1331.5932 ~  seconds
% FISTA / alpha = 0.40 (with + one L22):     1380.5802 ~  seconds
% FISTA / alpha = 0.40 (without):   1302.9458 ~  seconds

% CD : Alpha = 0.95 3998.9883 seconds
% CD: Alpha = 0.40 773.8899 seconds



%% Results
% With the regularization path
%   reg_path = [1:-0.01:0.9,...
    %0.895:-0.005:0.30,...
    % 0.2975:-0.0025:0.05];
% we have the following timings (averaged over 5 runs;

% nPDHG with EN and alpha = 0.95 (5 runs): 365.55 seconds
% nPDHG with EN and alpha = 0.40 (5 runs): 113.53 seconds
%
% FISTA with EN and alpha = 0.95 (5 runs): 4208.01 seconds
% FISTA with EN and alpha = 0.40 (5 runs): 1407.22 seconds
%
% CD with EN and alpha = 0.95 (5 runs): 
% CD with EN and alpha = 0.40 (5 runs): 1018.73 seconds


% nPDHG with NOGL (5 runs): 278.14 seconds
% FISTA with NOGL (5 runs): 3036.38 seconds

% nPDHG with linf (5 runs): 289.98 seconds
% FISTA with linf (5 runs): 2534.65 seconds


%% Main code
% Load the data
new_run = 1;
nb_runs = 5; % Number of runs + total time

if(new_run)
    [amat_annual,pprior,pempirical,Ed,n0,n1,name_features,...
        ind_nan_months,groups] = prepare_wildfire_data;
    m = length(Ed);     % Number of features
end

% Run the script (with new_run = false)
tic
for i=1:1:nb_runs
    script_maxent_inf
end
time_total = toc;

disp(' ')
disp('%%%%%%%%%%%%%%%%%%%%')
disp(["Average time over ", num2str(nb_runs), ' runs is: ',  num2str(time_total/nb_runs)])
disp('%%%%%%%%%%%%%%%%%%%%')