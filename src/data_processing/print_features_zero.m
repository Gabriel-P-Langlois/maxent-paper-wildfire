%% FUNCTION
% This function prints the features that were found to be significant,
% i.e., a feature whose corresponding dual component w_{i} ~= 0.
%
% Input
%
%   sol_w_lambda:   mx1 vector of the dual solution at the hyperparameter
%                   lambda
%                   w_{i} ~= 0 --> i^{th} feature is significant
%   name_features:  mx1 string containing the names of the features
%   groups:         2x5 cell structure containing the indices of the groups
%           The first row contains the names of the groups
%           The second row contains the indices of the groups

function print_features_zero(sol_w_lambda,name_features,groups)
ind_zero = (sol_w_lambda == 0).*(1:1:length(sol_w_lambda)).';

% Array for variable group names
groups_alt = {'Fire', 'Antecedent', 'Vegetation', 'Human', 'Topography'};

% Identify features in groups that are found to be nonzero.
num_groups = length(groups);
for g=1:1:num_groups
    ind_found = intersect(ind_zero,groups{2,g});
    if(~isempty(ind_found))
        disp(' ')
        disp(['Features in the group ',groups_alt{g}, ' that were found to be zero:'])
        disp(name_features(ind_found))
    end
end
end