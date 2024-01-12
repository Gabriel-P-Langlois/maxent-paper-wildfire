function [amat_annual,pprior,pempirical,Ed,n0,n1,name_features,...
    idx_features,ind_nan_mths,groups] = prepare_wildfire_data(use_quadratic_features)


%% Input
% use_quadratic_features: Boolean variable. If set to true, the features
% will be augmented to include quadratic products. E.g.,
%   "VPD         x Ant_RH     "
%   "Tmax_max3   x Forest      "
%   "Tmax        x Forest      "


%% Data extraction
% Extract the features (as a matrix) and their names (as a vector)
amat = [h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block0_values')', ...
    h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block1_values')'];
name_features = [h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block0_items')',...
    h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block1_items')'].';

% Remove Solar, Elev, RH, Ant_RH, CAPE Ant_Tmax, FFWI_max3, Delta_T
% Avgprec_3mo, AvgVPD_4mo, and RH_min3 features.

amat(:,[4,6,7,9,11,12,13,14,30,35,38]) = [];
name_features([4,6,7,9,11,12,13,14,30,35,38]) = [];

% Groups extraction

groups{1,1} = name_features([1,2,3,4,5,7,8,14,15,16,17,18,19,23,24,27]);
groups{1,2} = name_features([6,12,13,22,25,28,32]);
groups{1,3} = name_features([26,29,30,35]);
groups{1,4} = name_features([9,10,11,31,33,34]);
groups{1,5} = name_features([20,21]);

groups{2,1} = [1,2,3,4,5,7,8,14,15,16,17,18,19,23,24,27];   % fire 
groups{2,2} = [6,12,13,22,25,28,32];                        % Antecedants
groups{2,3} = [26,29,30,35];                                % Vegetation
groups{2,4} = [9,10,11,31,33,34];                           % humans
groups{2,5} = [20,21];                                      % Topography

% Store indices of relevant features.
idx_features = (1:1:length(name_features)).';

% Extract information, including presence of fire and regional indices.
data_info = h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block2_values')';

% Discard NaN values.
ind_nan = isnan(amat(:,1));
amat = amat(~ind_nan,:);
data_info = data_info(~ind_nan,:);

% Extract information, including presence of fire and regional indices.
nb_months = data_info(end,2) - data_info(1,2) + 1;
nb_spatial_points = length(data_info(:,1))/nb_months;
nb_years = nb_months/12;

fire_indicator = int64(zeros(length(data_info(:,1))/nb_years,1));
reg_index = data_info(:,3);


%% Data processing I: Annual averaging and features preparation
% Take the annual average of the wildfire data set and 
% compute grid cells where at least one fire occured.
amat_annual =  zeros(length(data_info(:,1))/nb_years,length(amat(1,:)));
for i=1:1:nb_months
    j = 1 + rem(i-1,12);
    range = (1 + (j-1)*nb_spatial_points):(j*nb_spatial_points);
    amat_annual(range,:) = amat_annual(range,:) + ... 
        amat((1 + (i-1)*nb_spatial_points):(i*nb_spatial_points),:);
    fire_indicator(range) = fire_indicator(range) + ...
        data_info((1 + (i-1)*nb_spatial_points):(i*nb_spatial_points),1);
end
amat_annual = amat_annual/double(nb_years);

% Separate data with fires and those with no fire.
ind_fire_yes = (fire_indicator >= 1);

% Compute the number of background and presence-only data points
n0 = length(ind_fire_yes(~ind_fire_yes));
n1 = length(ind_fire_yes(ind_fire_yes));

% Augment the annual matrix with product features if desired
if(use_quadratic_features)
    m0 = size(amat_annual,2);
    amat_annual = [amat_annual,zeros(n0+n1,(m0+1)*m0/2)];
    name_features = [name_features;zeros((m0+1)*m0/2,1)];
    counter = 1;
    for i=1:1:m0
        for j = i:1:m0
            amat_annual(:,m0+counter) = amat_annual(:,i).*amat_annual(:,j);
            name_features(m0+counter) = name_features(i) + " x " + name_features(j); 
        counter = counter + 1;
        end
    end
end

% Scale the features
max_amat_annual = max(amat_annual);
min_amat_annual = min(amat_annual);
amat_annual = (amat_annual - min_amat_annual)./...
    (max_amat_annual-min_amat_annual);

% Select subarray of regional indices for the first year only
reg_index_annual = reg_index(1:(12*nb_spatial_points));

clear amat range fire_indicator reg_index


%% Data processing II: Compute the prior distribution
pprior = ncread('pred_fire_masked_prob_all_mons.h5',"/df/block0_values").';

% Remove nan value
% Note: nan values grid cells correspond to gridcells in the ocean.
ind_nan_mths = isnan(pprior);

% Flattens the prior distribution to a single row
pprior = pprior(~ind_nan_mths); 

% Set gridcells that did not observe a fire to an insignificant 
% but nonzero probablity.
pprior(pprior == 0) = min(pprior((pprior ~= 0)))/10;
pprior = pprior/sum(pprior);


%% Data processing III: Compute the empirical distribution
% Note: region labeled as 0 in the data is region labeled as 1 in the code.
% n1_r == nb of grid points with a fire within the region i.

r = double(max(reg_index_annual)+1);
nr = zeros(r,1);
pempirical_r = zeros(r,1); pempirical = zeros(n0+n1,1); sum_pempirical = 0;

% Compute empirical probabilities per region then normalize
for i=1:r
    nr(i) = double(sum(reg_index_annual == i-1));
    n1_r = length(unique(data_info(and(ind_fire_yes, (reg_index_annual == i-1)),[2,4:5]),'rows'));
    pempirical_r(i) = (1/(nr(i)));
    sum_pempirical = sum_pempirical + pempirical_r(i)*n1_r;
end
pempirical_r = pempirical_r/sum_pempirical;

for i=1:r
    pempirical(and(reg_index_annual == i-1,ind_fire_yes),:) = pempirical_r(i);
end

% Compute empirical features
Ed = amat_annual'*pempirical;
end


%% Information regarding the wildfire data set.
% block0_values: 38 Features (see bottom of the script)
% block1_values: 7 Features (see bottom of the script)
% block2_values: 5 quantities (see bottom of the script)


% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block0_items')
% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block1_items')
% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block2_items')