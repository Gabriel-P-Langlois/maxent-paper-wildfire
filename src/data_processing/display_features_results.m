function display_features_results(sol_w,lambda,name_features,groups)
% This function display various results for the feature selection problem
%
% Input
%
%   sol_w:   l_max x m matrix of the dual solution. Here, l_max = nb of
%       points in the regularization path and w_{i} ~= 0 --> i^{th} feature 
%       is significant
%   lambda: l_max x 1 vector containing the hyperparameters. Here, 
%           lambda(1) == smallest hyperparameter for which sol_w(:,1) == 0.
%   name_features:  mx1 string containing the names of the features
%   groups:         2x5 cell structure containing the indices of the groups
%           The first row contains the names of the groups
%           The second row contains the indices of the groups


% For each group, find the first (largest) hyperparameter for which a
% nonzero feature appears.
num_groups = length(groups);
m = length(sol_w(:,1));
ind_instance_groups  = identify_group_thresholds(sol_w,lambda,groups);


% Display results I: First hyperparameter to appear in each group
disp(" ");
for g=1:1:num_groups
    disp(['The largest hyperparameter that results in a nonzero feature in the ', ...
        groups{1,g}, ' group is equal to ', num2str(lambda(ind_instance_groups(g))/lambda(1)),' lambda_est.'])
    disp(['The feature(s) found in the ', groups{1,g}, 'group at that hyperparameter is or are:'])
    disp(name_features(intersect((sol_w(:,ind_instance_groups(g)) ~= 0).*(1:1:m).',groups{2,g})))
    disp(" ")
end
disp("----------")


% Display results II: Nonzero features found each time a feature from 
% another group is discovered, in order in which they appear
sorted_ind = sort(ind_instance_groups);
for g=1:1:num_groups
    disp(['At the hyperparameter ',num2str(lambda(sorted_ind(g))/lambda(1)), ...
        ' lambda_est, the following nonzero features were found:'])
    print_features_nonzero(sol_w(:,sorted_ind(g)),name_features,groups)
    disp(' ')
    disp("-----")
end
disp("----------")


% Display results III: Print zero features found at the end of the
% regularization path
disp("The following features were found to be zero at the end of the regularization path:")
print_features_zero(sol_w(:,end),name_features,groups)



% Display results IV: Print nonzero features found at the end of the
% regularization path
disp("The following features were found to be zero at the end of the regularization path:")
print_features_nonzero(sol_w(:,end),name_features,groups)

%% END
end