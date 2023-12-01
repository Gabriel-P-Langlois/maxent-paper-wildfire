function ind_instance_groups = identify_group_thresholds(sol_w,lambda,groups)
% This function finds the first (largest) hyperparameter for which a
% nonzero feature appears in each group.
%
% Input
%
%   sol_w:   l_max x m matrix of the dual solution. Here, l_max = nb of
%       points in the regularization path and w_{i} ~= 0 --> i^{th} feature 
%       is significant
%   lambda: l_max x 1 vector containing the hyperparameters. Here, 
%           lambda(1) == smallest hyperparameter for which sol_w(:,1) == 0.
%   groups:         2x5 cell structure containing the indices of the groups
%           The first row contains the names of the groups
%           The second row contains the indices of the groups


reg_path_length = length(lambda);
num_groups = length(groups);
m = length(sol_w(:,1));

first_instance_found = false(num_groups,1);
ind_instance_groups = zeros(num_groups,1);

% For each group, find the first (largest) hyperparameter for which a
% nonzero feature appears.
for g=1:1:num_groups
    for j=1:1:reg_path_length
        ind_nonzero = (sol_w(:,j) ~= 0).*(1:1:m).';
        first_instance_found(g) = ~isempty(intersect(ind_nonzero,groups{2,g}));
        if(first_instance_found(g))
            ind_instance_groups(g) = j;
            break;
        end
    end
end