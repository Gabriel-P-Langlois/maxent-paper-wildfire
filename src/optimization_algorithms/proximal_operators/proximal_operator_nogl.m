%%  Description
%   This function computes the proximal operator
%   of the non-overlapping group lasso penalty. That is, we compute
%
%   yprox = argmin_{y \in \Rn} (0.5/t)*norm(x-y)_{2}^{2} + 
%               \sum_{g=1}^{G} t*\sqrt(m_{g}}\normtwo_{2,g}{y};
%
%   where x \in \Rm, t > 0, and m_{g} = # of elements in the group g.
%
% Note: The input groups consist of a cell structure containing the 
%       indices of the groups. The groups should form a partition 
%       of {1,...,m} and should not overlap.


%% Function
function yprox = proximal_operator_nogl(x,t,groups)
    yprox = x;
    for i=1:1:length(groups)
        ind = groups{i};
        yprox(ind) = proximal_operator_l2(x(ind),sqrt(length(ind))*t);
    end
end