%% FUNCTION: Extract and process the data from the wildfire data set
% Note: Jatan and I double-checked that the prior distribution he plots 
% corresponds exactly to the prior distribution we use for pprior here.

function [amat_annual,pprior,pempirical,Ed,n0,n1,name_features,idx_features,ind_nan_mths] = process_augment_wildfire_data


%% Data extraction
% Extract the features and their names
amat = [h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block0_values')', ...
    single(h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block1_values'))'];
name_features = [h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block0_items')',h5read('clim_fire_freq_12km_w2020_data.h5', '/df/block1_items')'].';

% Remove Solar, Elev, Ant_Tmax, and Delta_T features.
% Those correspond to entries 4, 6, 9, and 35.
amat(:,[4,6,9,35]) = [];
name_features([4,6,9,35]) = [];

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


% Take the annual average of the wild fire data set and 
% compute grid cells where at least one fire occured.
amat_annual =  single(zeros(length(data_info(:,1))/nb_years,length(amat(1,:))));
for i=1:1:nb_months
    j = 1 + rem(i-1,12);
    range = (1 + (j-1)*nb_spatial_points):(j*nb_spatial_points);
    amat_annual(range,:) = amat_annual(range,:) + ... 
        amat((1 + (i-1)*nb_spatial_points):(i*nb_spatial_points),:);
    fire_indicator(range) = fire_indicator(range) + ...
        data_info((1 + (i-1)*nb_spatial_points):(i*nb_spatial_points),1);
end
amat_annual = amat_annual/single(nb_years);

% Separate data with fires and those with no fire.
ind_fire_yes = (fire_indicator >= 1);

% Compute the number of background and presence-only data points
n0 = length(ind_fire_yes(~ind_fire_yes));
n1 = length(ind_fire_yes(ind_fire_yes));


% Augment the annual matrix with product features
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

% Scale the features
max_amat_annual = max(amat_annual);
min_amat_annual = min(amat_annual);
amat_annual = (amat_annual - min_amat_annual)./...
    (max_amat_annual-min_amat_annual);

% Select subarray of regional indices for the first year only
reg_index_annual = reg_index(1:(12*nb_spatial_points));

clear amat range fire_indicator reg_index


%% Algorithm preparation I: Compute the prior distribution
pprior = ncread('pred_fire_masked_prob_all_mons.h5',"/df/block0_values").';

% Remove nan value
% Note: nan values grid cells correspond to gridcells in the ocean.
ind_nan_mths = isnan(pprior);
size(pprior)
size(ind_nan_mths)

% Flattens the prior distribution to a single row
pprior = pprior(~ind_nan_mths); 
size(pprior)

% Set gridcells that did not observe a fire to a nonzero but insignificant
% probablity.
pprior(pprior == 0) = min(pprior((pprior ~= 0)))/10;
pprior = single(pprior/sum(pprior));


%% Algorithm preparation II: Compute the empirical distribution
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
pempirical = single(pempirical);

% Compute empirical features
Ed = amat_annual'*pempirical;
end



%% Information regarding the wildfire data set.
% block0_values: 38 Features (see bottom of the script)
% block1_values: 7 Features (see bottom of the script)
% block2_values: 5 quantities (see bottom of the script)


% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block0_items')
% 
% ans =
% 
%   38×1 cell array
%     {'Tmax       '}
%     {'VPD        '}
%     {'Prec       '}
%     {'Solar      '}
%     {'Wind       '}
%     {'Elev       '}
%     {'RH         '}
%     {'FM1000     '}
%     {'Ant_Tmax   '}
%     {'AvgVPD_3mo '}
%     {'Avgprec_3mo'}
%     {'Ant_RH     '}
%     {'CAPE       '}
%     {'FFWI_max3  '}
%     {'FFWI_max7  '}
%     {'Tmin       '}
%     {'Camp_dist  '}
%     {'Camp_num   '}
%     {'Road_dist  '}
%     {'Avgprec_4mo'}
%     {'Avgprec_2mo'}
%     {'VPD_max3   '}
%     {'VPD_max7   '}
%     {'Tmax_max3  '}
%     {'Tmax_max7  '}
%     {'Tmin_max3  '}
%     {'Tmin_max7  '}
%     {'Slope      '}
%     {'Southness  '}
%     {'AvgVPD_4mo '}
%     {'AvgVPD_2mo '}
%     {'SWE_mean   '}
%     {'SWE_max    '}
%     {'AvgSWE_3mo '}
%     {'Delta_T    '}
%     {'Biomass    '}
%     {'Lightning  '}
%     {'RH_min3    '}

% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block1_items')
% 
% ans =
% 
%   7×1 cell array
% 
%     {'Antprec_lag1'}
%     {'Forest      '}
%     {'Grassland   '}
%     {'Urban       '}
%     {'Antprec_lag2'}
%     {'Popdensity  '}
%     {'Housedensity'}

% h5read('clim_fire_freq_12km_w2020_data.h5','/df/block2_items')
% 
% ans =
% 
%   5×1 cell array
% 
%     {'fire_freq'}
%     {'month    '}
%     {'reg_indx '}
%     {'X        '}
%     {'Y        '}