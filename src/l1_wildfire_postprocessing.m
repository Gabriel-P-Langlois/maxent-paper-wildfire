%% WIP

function idx_lambda_found = l1_wildfire_postprocessing(sol_npdhg_w,reg_path,name_features)

% Example 1: Sparsity_plot
% Count the number of nonzero elements at every hyperparameter and plot
l_max = length(reg_path);
sparsity_plot = zeros(l_max,1);
for j=1:1:l_max
    sparsity_plot(j) = sum((sol_npdhg_w(:,j)~= 0));
end
plot(reg_path,sparsity_plot,'*')
xlabel('\lambda/\lambda_{max}')
ylabel('Nb of non-zero coefficients')
set ( gca, 'xdir', 'reverse' )
title('Sparsity plot')


% Example 2: Display for which lambdas where the features discovered
disp('New features discovered at the following values of lambda_max: ')
disp(' ')
idx_lambda_found = [];
for j=1:1:l_max
    i_target = find(sparsity_plot == j,1);
    idx_lambda_found = [idx_lambda_found;i_target];
end
idx_lambda_found = unique(idx_lambda_found,'stable');
display((reg_path(idx_lambda_found)/reg_path(1)).');

% Example 3: Display indices of discovered features
idx_features_found = [];
for j=1:length(idx_lambda_found)
    tmp = find(sol_npdhg_w(:,idx_lambda_found(j))~=0);
    idx_features_found = [idx_features_found;tmp];
    %disp(find(sol_npdhg_w(:,idx_lambda_found(j))~=0))
end
idx_features_found = unique(idx_features_found,'stable');
disp(name_features(idx_features_found))


% UNSORT IF POSSIBLE

% nb_features = max(sparsity_plot);
% features_ranked = zeros(nb_features,1);
% for j=1:1:nb_features
%     i_target = find(sparsity_plot == j,1);
%     ind_target = find(sol_npdhg_w(:,i_target) ~= 0);
%     features_ranked(j,1) = setdiff(ind_target,features_ranked);
%     if(j < 10)
%         disp([num2str(j),'.   ',num2str(name_features(features_ranked(j,1)))])
%     else
%         disp([num2str(j),'.  ',num2str(name_features(features_ranked(j,1)))])
%     end
% end
disp(' ')
end