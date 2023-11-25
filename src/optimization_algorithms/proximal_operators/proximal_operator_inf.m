%%  Description
%   This function computes the proximal operator
%   of the linf penalty. That is, we compute
%
%   yprox = argmin_{y \in \Rn} (0.5/t)*norm(x-y)_{2}^{2} + norminf(y)
%
%   where x \in \Rn, t > 0.
%
%   Note: Thanks to Moreau's decomposition theorem, computing this prox
%   point is equivalent to projection x on the l1 ball of radius t.
%
%   The code below is adapted from Laurent Condat's code (2016).

%% Function
function yprox = proximal_operator_inf(x,t)
yprox = max(abs(x)-max(max((cumsum(sort(abs(x),1,'descend'),1)-t)...
    ./(1:size(x,1))'),0),0).*sign(x);
yprox = x - yprox;
end