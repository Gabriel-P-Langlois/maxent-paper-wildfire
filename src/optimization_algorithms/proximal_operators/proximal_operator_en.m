%%  Description
%   This function computes the proximal operator
%   of the elastic net penalty. That is, we compute
%
%   yprox = argmin_{y \in \Rn} (0.5/t)*norm(x-y)_{2}^{2} + alpha*normone(y)
%                   + 0.5*(1-alpha)*norm(y)_{2}^{2}
%
%   where x \in \Rn, t > 0 and alpha \in [0,1].
%
%   Note 1: alpha = 0 yields Tikhonov regularization (with identity matrix)
%           alpha = 1 yields the soft thresholding operator
%
%   Note 2: For alpha \in (0,1), we can write the minimization problem as
%
%   argmin_{y \in \Rn} (0.5*(1+t*(1-alpha))/t)*norm(x/(1+t*(1-alpha)) -
%   y)_{2}^{2} + alpha*normone(y).
%
%   Hence the prox is weighted soft thresholding.


%% Function
function yprox = proximal_operator_en(x,t,alpha)

% Reduce the problem to soft thresholding by calculating xeff and teff s.t.
% yprox = argmin_{y} 0.5*normsq{xeff-y} + teff*normone(y).
tmp = 1/(1+t*(1-alpha));
xeff = x*tmp;
teff = alpha*t*tmp;

% Soft thresholding operation
yprox = abs(xeff)-teff;
yprox = sign(xeff).*(yprox+abs(yprox))*0.5;

end