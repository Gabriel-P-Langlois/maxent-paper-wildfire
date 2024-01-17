%%  Description
%   This function computes the proximal operator of the l2 penalty.
%
%   yprox = argmin_{y \in \Rn} (0.5/t)*norm(x-y)_{2}^{2} + normtwo(y)
%
%   where x \in \Rn and t > 0.
% 
%   We use Moreau's decomposition for this and instead project the
%   point x onto the l2 ball of radius t

%% Function
function yprox = proximal_operator_l2(x,t)

yprox = max(0,1-t/norm(x))*x;

end