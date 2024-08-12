function reshaped_sol_p = reshape_probability(sol_p,ind_nan_months,l_max)
% RESHAPED_SOL_P
% This function appends the probability vector with the missing
% nan values that were discarded in the prepare_wildfire_data.m 
% function. This is required to use the python script prob_plot_script.py
%
% Input:
%   - sol_p: An l_max x N matrix of the primal solution. Here, l_max = nb 
%            of points in the regularization path.
%   - ind_nan_months: Grid cells in the wildfire data to ignore (correspond
%                     to gridcells in the ocean (provided by the function
%                     prepare_wildfire_data).
%   - l_max: A scalar equal to the length of the regularization path
%
% Output:
%   - reshaped_sol_p: A reshaped vector of primal solution sol_
%                     with the missing nan values for the python script
%                     prob_plot_script.py.

% Quantities
nan_ind = ind_nan_months(:);
n_ind = length(nan_ind);
n_solp = size(sol_p,1);
n_diff = n_ind - n_solp;

reshaped_sol_p = [sol_p;zeros(n_diff,l_max)];

% Set NaN and non-NaN values
for k=1:1:l_max
    reshaped_sol_p(~nan_ind,k) = reshaped_sol_p(1:n_solp,k);
    reshaped_sol_p(nan_ind,k) = nan;
end
end