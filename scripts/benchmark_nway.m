function benchmark_nway(inDir, outDir, rank, nStart, maxIt, engine)
% BENCHMARK_NWAY  PARAFAC of the Ombrone EEMs with drEEM's PARAFAC engines.
%
%   benchmark_nway('outputs/benchmark/input', 'outputs/benchmark/nway', 3, 20, 20000, 'nway')
%   benchmark_nway('outputs/benchmark/input', 'outputs/benchmark/parafac3w', 3, 20, 20000, 'parafac3w')
%
% Input: eemkit_corrected.csv written by scripts/benchmark_export.py, i.e.
% eemkit's corrected EEMs of the samples in the final model (scatter = NA),
% so the comparison tests the PARAFAC fit only.
%
% Engines, called as drEEM 2 (fitparafac) calls them:
%   'nway'       the N-way toolbox parafac (Andersson & Bro 2000), drEEM's
%                engine up to version 1 and its option toolbox="nway";
%                in drEEM 2 it is nway.models.parafac, otherwise the
%                stand-alone toolbox's parafac (File Exchange #1088);
%   'parafac3w'  parafac3w, drEEM 2's default engine for non-negative models
%                (its own defaults except ConvCrit and MaxIt).
%
% Method, chosen to match eemkit's final model:
%   * each sample divided by its Frobenius norm (sum over observed values),
%     scores multiplied back afterwards (Murphy et al. 2013);
%   * non-negativity on all three modes;
%   * nStart random starts (rng(k) for start k), convergence 1e-8, at most
%     maxIt iterations; the start with the lowest sum of squared errors is
%     kept; a start that used every iteration counts as not converged and is
%     only used if no start converged (as drEEM does);
%   * missing values handled by the engine.
%
% Output (same files as scripts/benchmark_stardom.R, read by
% scripts/benchmark_compare.py --engine nway | parafac3w):
%   pf_ex_loadings.csv, pf_em_loadings.csv  unit-norm loadings
%   pf_fmax.csv                             Fmax per sample, R.U.
%   pf_info.csv                             fit statistics and versions

if nargin < 3, rank = 3; end
if nargin < 4, nStart = 20; end
if nargin < 5, maxIt = 20000; end
if nargin < 6, engine = 'nway'; end
engine = char(engine);
ctol = 1e-8;

hasPkg = ~isempty(which('nway.models.parafac'));
switch engine
    case 'nway'
        if hasPkg
            fit = @(X, k) nway.models.parafac(X, k, [ctol 2 0 0 NaN maxIt], [2 2 2]);
            label = "N-way toolbox parafac (drEEM " + dreemVersion() + ")";
        elseif exist('parafac', 'file') == 2
            fit = @(X, k) parafac(X, k, [ctol 2 0 0 NaN maxIt], [2 2 2]);
            label = "N-way toolbox parafac";
        else
            error(['N-way toolbox not found: install drEEM 2 (Add-Ons) or put the ' ...
                   'N-way toolbox (File Exchange #1088) on the MATLAB path.']);
        end
    case 'parafac3w'
        if exist('parafac3w', 'file') ~= 2 || ~hasPkg
            error('parafac3w not found: install drEEM 2 (Add-Ons).');
        end
        fit = @(X, k) fit3w(X, k, ctol, maxIt);
        label = "drEEM parafac3w (drEEM " + dreemVersion() + ")";
    otherwise
        error('engine must be ''nway'' or ''parafac3w''');
end
if ~exist(outDir, 'dir'), mkdir(outDir); end

file = fullfile(inDir, 'eemkit_corrected.csv');
% the header is set explicitly: with numeric column names (excitation
% wavelengths) MATLAB would otherwise read it as data
opts = detectImportOptions(file, 'TreatAsMissing', 'NA', 'NumHeaderLines', 1);
opts.VariableNamesLine = 1;
opts.DataLines = [2 Inf];
opts.VariableNamingRule = 'preserve';
opts = setvartype(opts, 'string');
opts = setvartype(opts, 2:numel(opts.VariableNames), 'double');
T = readtable(file, opts);
names = T.Properties.VariableNames;
ex = str2double(names(3:end));
samples = unique(T.sample, 'stable');
em = T.em(T.sample == samples(1));
nS = numel(samples); nEx = numel(ex); nEm = numel(em);
X = nan(nS, nEx, nEm);
for i = 1:nS
    rows = T(T.sample == samples(i), :);
    if ~isequal(rows.em, em), error('emission grid differs for %s', samples(i)); end
    X(i, :, :) = permute(table2array(rows(:, 3:end)), [3 2 1]);
end

scale = sqrt(sum(X.^2, [2 3], 'omitnan'));
Xn = X ./ scale;
obs = ~isnan(Xn);

Fs = cell(nStart, 1); errs = nan(nStart, 1); its = nan(nStart, 1);
t0 = tic;
for k = 1:nStart
    rng(k);
    [F, it] = fit(Xn, rank);
    Fs{k} = F; its(k) = it;
    errs(k) = sum((Xn(obs) - reconstruct(F, obs)).^2);
    fprintf('start %2d: %6d iterations, SSE %.10g\n', k, it, errs(k));
end
seconds = toc(t0);

converged = its < maxIt;
pick = errs;
if any(converged), pick(~converged) = Inf; end
[~, kb] = min(pick);

A = Fs{kb}{1}; B = Fs{kb}{2}; C = Fs{kb}{3};
nb = vecnorm(B); nc = vecnorm(C);
B = B ./ nb; C = C ./ nc; A = A .* (nb .* nc);
scores = A .* scale(:);                       % back to R.U.
fmax = scores .* max(B, [], 1) .* max(C, [], 1);

rsq = 1 - errs(kb) / sum(Xn(obs).^2);
sameMin = sum(errs <= errs(kb) * (1 + 1e-3));
% two arguments, as drEEM's fitparafac calls it (missing cells left out)
if hasPkg
    cc = nway.modeleval.corcond(Xn, {A, B, C});
else
    cc = corcond(Xn, {A, B, C});
end

comp = "C" + string(1:rank);
writetable(array2table([ex(:) B], 'VariableNames', ["ex" comp]), ...
           fullfile(outDir, 'pf_ex_loadings.csv'));
writetable(array2table([em(:) C], 'VariableNames', ["em" comp]), ...
           fullfile(outDir, 'pf_em_loadings.csv'));
writetable([table(samples, 'VariableNames', "sample") array2table(fmax, 'VariableNames', comp)], ...
           fullfile(outDir, 'pf_fmax.csv'));
v = ver('MATLAB');
info = table(rank, nStart, maxIt, ctol, rsq, its(kb), sum(converged) / nStart, ...
             sameMin / nStart, cc, seconds, string(v.Version), ...
             'VariableNames', ["rank" "nstart" "maxit" "ctol" "rsq" "iter" ...
             "converged_fraction" "same_minimum_fraction" "corcondia" "seconds" "version"]);
info.engine = label + ", MATLAB " + string(v.Version);
writetable(info, fullfile(outDir, 'pf_info.csv'));
disp(info);
end

function [F, it] = fit3w(X, k, ctol, maxIt)
% parafac3w as drEEM 2 calls it: only ConvCrit and MaxIt are set.
res = parafac3w(X, k, struct('ConvCrit', ctol, 'MaxIt', maxIt));
F = {res.A, res.B, res.C};
it = numel(res.allfits);
end

function M = reconstruct(F, obs)
model = zeros(size(obs));
for f = 1:size(F{1}, 2)
    model = model + F{1}(:, f) .* reshape(F{2}(:, f), 1, []) .* reshape(F{3}(:, f), 1, 1, []);
end
M = model(obs);
end

function s = dreemVersion()
s = "not installed";
a = matlab.addons.installedAddons;
i = find(a.Name == "drEEM toolbox", 1);
if ~isempty(i), s = string(a.Version(i)); end
end
