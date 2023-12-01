%%  Description
%   This function computes the proximal operator of the linf penalty.
%
%   yprox = argmin_{y \in \Rn} (0.5/t)*norm(x-y)_{2}^{2} + norminf(y)
%
%   where x \in \Rn and t > 0.
%
%   Note: This uses the fast implementation [...]


%% Function
function yprox = proximal_operator_inf(x,t)

yprox = 0;

end