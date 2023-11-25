function stop = convergence_criterion_nogl(num_iters,max_iters,...
                t,groups,diff,tol)
    stop = 0;

    if(num_iters >= 40)
        stop = 1;
        G = length(groups);
        for g=1:G
            ind = groups{g};
            l = sqrt(length(ind));
            stop = stop*(norm(diff(ind),2) <= l*t*(1+tol));
        end
    end

    stop = stop || (num_iters >= max_iters);
end