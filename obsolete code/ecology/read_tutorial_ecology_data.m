%% Script for handling the ecological data
% There are no duplicates in the tutorial data


%% Data extraction
% Read background samples
T_background = readtable('background.csv');
amat = T_background{:,4:end}; % Feature matrix
n0 = size(amat,1);

% Read presence-only data
T_presence = readtable('bradypus_swd.csv');
amat_presence = T_presence{:,4:end};
n1 = size(amat_presence,1);

% Concatenate the two
amat = [amat_presence;amat];
clear amat_presence T_background T_presence


%% Data preparation
% Create new features
% TBC.

% Scale the features
max_amat = max(amat);
min_amat = min(amat);
amat = (amat - min_amat)./...
    (max_amat-min_amat);

% Extract empirical features
Ed = amat(1:n1,:);

% Define the prior
pprior = ones(n0+n1,1)/(n0+n1);


%% Code for checking duplicated coordinates
% T_species = readtable('bradypus_swd.csv');
% coord_species = T_species{:,2:3};
% 
% T = readtable('background.csv');
% coord_background = T{:,2:3};
% full_coord = [coord_species;coord_background];
% 
% [u,~,~] = unique(full_coord, 'rows', 'first');
% hasDuplicates = size(u,1) < size(full_coord,1); % = 0: No duplicates!