%%  Description
%   This function computes the proximal operator
%   of the non-overlapping group lasso penalty. That is, we compute
%
%   yprox = argmin_{y \in \Rn} (0.5/t)*norm(x-y)_{2}^{2} + 
%               [...]
%
%   where x \in \Rn, t > 0, and G is a non-overlapping group partition.
%
%   Since the G groups are non-overlapping, we can divide the problem
%   into G subproblems and solve each subproblem using the l2 proximal
%   operator.


%% Function
function yprox = proximal_operator_nogl(x,t,G)

% Use the partition G to separate x into G distinct group
%
% [...]
%
% Apply the l2 prox on each group
%
% [...]
%
% Combine the results and return the proximal operator

yprox = 0;
end