%% Short script to play around with the wildfire data set.
% Play around with the wildfire dataset, maybe look at different
% probability distributions...


% Read the data
A = [h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block0_values')',single(h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block1_values'))'];
data8 = h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block2_values')';


% Discard NaN values
ind_nan = isnan(A(:,1));
A = A(~ind_nan,:);
data8 = data8(~ind_nan,:);


% Obtain regional indices
reg_index = data8(:,3);
ind_r_zero = (reg_index == 0);


% Discard data whose regional index is 0.
reg_index(reg_index == 0) = [];
A = A(~ind_r_zero,:);
data8 = data8(~ind_r_zero,:);


% Separate data with fires and those with no fire.
ind_fire_yes = (data8(:,1) >= 1);


% Compute the number of background and presence-only data points
n0 = length(unique(data8(~ind_fire_yes,[2,4:5]),'rows'));
n1 = length(unique(data8(ind_fire_yes,[2,4:5]),'rows'));
n = n0 + n1;


% Scale the features -- This will be discussed in the numerics section.
max_A = max(A);
min_A = min(A);
A = (A - min_A)./(max_A-min_A);


% Compute priors for the different regions + prior for the algorithm
r = double(max(reg_index));
nr = zeros(r,1);
prior_r = zeros(r,1);
pprior = zeros(n,1);
for i=1:r
    nr(i) = double(sum(reg_index == i));
    prior_r(i) = (1/(nr(i)*r));
    pprior(reg_index == i) = prior_r(i);
end

% % Define the empirical distribution I: absence grid probability is set to 0
% Leeway for lambda, but 0 probability is bad for absence point
% underweighted!
% pempirical_r = zeros(r,1); pempirical = zeros(n,1); sum_pempirical = 0;
% for i=1:r
%     n1_r = length(unique(data8(and(ind_fire_yes, (reg_index == i)),[2,4:5]),'rows'));
%     pempirical_r(i) = (1/(nr(i)));
%     sum_pempirical = sum_pempirical + pempirical_r(i)*n1_r;
% end
% pempirical_r = pempirical_r/sum_pempirical;
% 
% for i=1:r
%     pempirical(and(reg_index == i,ind_fire_yes),:) = pempirical_r(i);
% end

% Define the empirical distribution II: Allow absence points to be nonzero
% NOTE: Lambda becomes small
% Overweighted the zeros! This is one extreme.

% Use something similar to the prior
pempirical_r = zeros(r,1); pempirical_r2 = zeros(r,1);
pempirical = zeros(n,1); sum_pempirical = 0;
for i=1:r
    n1_r = length(unique(data8(and(ind_fire_yes, (reg_index == i)),[2,4:5]),'rows'));
    pempirical_r(i) = (1/(nr(i)));
    pempirical_r2(i) = (1/(r*nr(i)));
    sum_pempirical = sum_pempirical + pempirical_r(i)*n1_r + pempirical_r2(i)*double(sum(and(~ind_fire_yes,reg_index == i)));
end
pempirical_r = pempirical_r/sum_pempirical;
pempirical_r2 = pempirical_r2/sum_pempirical;

for i=1:r
    pempirical(and(reg_index == i,ind_fire_yes),:) = pempirical_r(i);
    pempirical(and(reg_index == i,~ind_fire_yes),:) = pempirical_r2(i);
end

% Define expected value % pempirical = single(zeros(n,1)); pempirical(ind_fire_yes) = 1/n1;
Ed = A'*pempirical;

% Compute the smallest parameter for which the dual solution is zero.
lambda_est = norm(Ed - A'*pprior,inf);

% Compute hyperparameters to be used
reg_path = [1,0.95,0.90,0.85,0.80];
lambda = lambda_est*reg_path;

% Variable selection
alpha = lambda(2)/lambda(1);
s = alpha*pprior + (1-alpha)*pempirical;
lhs = abs(Ed - A'*pprior);    
rhs = lambda(1) - vecnorm(A,inf)'*sqrt(2*kldiv(s,pprior))/alpha;
ind = (lhs < rhs);         % Indices that are zero.

test = kldiv(s,pprior); disp(test)

% Display coefficients found to be zero.
m = length(Ed);         % Number of features
disp(['Percentage of coefficients found to be zero: ',num2str(100*sum(ind)/m)])


%% Auxiliary functions
% Kullback--Leibler divergence
function kl_num = kldiv(prob1,prob2)
    kl_num = prob1'*log(prob1./prob2);
end