function [ind,p_conv] = screening_en(lambda_plus,lambda,p_lambda,...
    p_empirical,alpha,Ed,amat,m)
    
    % Take a convex combination of the previous solution with the empirical
    % distribution, using the current and previous hyperparameters.
    beta = lambda_plus/lambda;
    p_conv = beta*p_lambda + (1-beta)*p_empirical;

    % Compute the lhs and rhs of the criterion and compare them.
    lhs = abs(Ed - amat.'*p_lambda);    

    % Requires verification
    rhs = alpha*lambda - ...
        ones(m,1)*sqrt(2*(p_conv.'*log(p_conv./p_lambda)))/beta;
    ind = (lhs >= rhs);
end