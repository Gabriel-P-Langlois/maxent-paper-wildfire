%% Function to reshape the probabilities, one by one.
function reshaped_sol_p = reshape_probability(sol_p,ind_nan_months,length_path)

% Quantities
nan_ind = ind_nan_months(:);
n_ind = length(nan_ind);
n_solp = size(sol_p,1);
n_diff = n_ind - n_solp;

reshaped_sol_p = [sol_p;zeros(n_diff,length_path)];

% Set NaN and non-NaN values
for k=1:1:length_path
    reshaped_sol_p(~nan_ind,k) = reshaped_sol_p(1:n_solp,k);
    reshaped_sol_p(nan_ind,k) = nan;
end

% Uncomment the line below if you want the format [data x months x length_path]
%reshaped_sol_p = reshape(reshaped_sol_p,[],12,length_path);
end