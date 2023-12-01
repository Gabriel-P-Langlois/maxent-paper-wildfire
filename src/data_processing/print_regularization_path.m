%% Function
% This function plots the number of nonzero feature as a function of the
% regularization path reg_path.
%
% Input
%
%   sol_w:   l_max x m matrix of the dual solution. Here, l_max = nb of
%   points in the regularization path
%                   w_{i} ~= 0 --> i^{th} feature is significant
%   lambda: l_max x 1 vector containing the hyperparameters. Here, 
%           lambda(1) == smallest hyperparameter for which sol_w(:,1) == 0.
%   groups:         2x5 cell structure containing the indices of the groups
%           The first row contains the names of the groups
%           The second row contains the indices of the groups

function  print_regularization_path(sol_w,lambda,groups)

reg_path = lambda/lambda(1); l_max = length(reg_path);

sparsity_plot = zeros(l_max,1);
for j=1:1:l_max
    sparsity_plot(j) = sum((sol_w(:,j)~= 0));
end

% For each group, find the first (largest) hyperparameter for which a
% nonzero feature appears.
num_groups = length(groups);
ind_instance_groups  = identify_group_thresholds(sol_w,lambda,groups);

% Plot the figure
figure(1)
plot(reg_path,sparsity_plot,'*')
xlabel('\lambda/\lambda_{max}')
ylabel('Nb of nonzero coefficients')
set ( gca, 'xdir', 'reverse' )
title('Sparsity plot')
ylim([0,35])

for g=1:1:num_groups
    xline(reg_path(ind_instance_groups(g)),'-',{['Feature from the ',groups{1,g},' group']});
end




end

