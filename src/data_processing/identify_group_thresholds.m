function ind_groups = identify_group_thresholds(sol_w,lambda,groups)
% IND_INSTANCE_GROUPS
% This finds the first (largest) hyperparameter for which a
% nonzero feature appears in each group.
%
% Input:
%   - sol_w: l_max x p matrix of the dual solution. Here, l_max = nb of
%            points in the regularization path and w_{i} ~= 0 --> 
%            i^{th} feature is significant.
%   - lambda: l_max x 1 vector containing the hyperparameters. Here, 
%           lambda(1) == smallest hyperparameter for which sol_w(:,1) == 0.
%   - groups: 2x5 cell structure containing the indices of the groups
%             The first row contains the names of the groups
%             The second row contains the indices of the groups
%
% Output:
%   - ind_groups: 5x1: A vector of five indices containing the first
%                 feature that appears in a group.


reg_path_length = length(lambda);
num_groups = length(groups);
p = length(sol_w(:,1));

first_instance_found = false(num_groups,1);
ind_groups = zeros(num_groups,1);

% For each group, find the first (largest) hyperparameter for which a
% nonzero feature appears.
for g=1:1:num_groups
    for j=1:1:reg_path_length
        ind_nonzero = (sol_w(:,j) ~= 0).*(1:1:p).';
        first_instance_found(g) = ...
            ~isempty(intersect(ind_nonzero,groups{2,g}));
        if(first_instance_found(g))
            ind_groups(g) = j;
            break;
        end
    end
end