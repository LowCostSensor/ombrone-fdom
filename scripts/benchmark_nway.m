function benchmark_nway(inDir, outDir, rank, nStart, maxIt)
% BENCHMARK_NWAY  PARAFAC of the Ombrone EEMs with the N-way toolbox (drEEM's engine).
%
%   benchmark_nway('outputs/benchmark/input', 'outputs/benchmark/nway', 3, 20, 20000)
%
% Input: eemkit_corrected.csv written by scripts/benchmark_export.py, i.e.
% eemkit's corrected EEMs of the samples in the final model (scatter = NA),
% so the comparison tests the PARAFAC fit only.
%
% Method, chosen to match eemkit's final model:
%   * each sample divided by its Frobenius norm (sum over observed values),
%     scores multiplied back afterwards (Murphy et al. 2013);
%   * non-negativity on all three modes (const = [2 2 2]);
%   * nStart random starts (rng(k) for start k), convergence 1e-8, at most
%     maxIt iterations; the start with the lowest sum of squared errors is kept;
%   * missing values handled by the N-way toolbox (expectation maximisation).
%
% Needs the N-way toolbox (Andersson & Bro 2000): it is bundled with drEEM
% 2, or free from MATLAB File Exchange (#1088). Its function `parafac` must be
% on the MATLAB path.
%
% Output (same files as scripts/benchmark_stardom.R, read by
% scripts/benchmark_compare.py --engine nway):
%   pf_ex_loadings.csv, pf_em_loadings.csv  unit-norm loadings
%   pf_fmax.csv                             Fmax per sample, R.U.
%   pf_info.csv                             fit statistics and versions

if nargin < 3, rank = 3; end
if nargin < 4, nStart = 20; end
if nargin < 5, maxIt = 20000; end
if exist('parafac', 'file') ~= 2
    error(['N-way toolbox function "parafac" not found on the MATLAB path. Add the ' ...
           'drEEM folder with its subfolders (addpath(genpath(...))) or install the ' ...
           'N-way toolbox (File Exchange #1088).']);
end
if ~exist(outDir, 'dir'), mkdir(outDir); end

T = readtable(fullfile(inDir, 'eemkit_corrected.csv'), 'TreatAsMissing', 'NA', ...
              'TextType', 'string', 'VariableNamingRule', 'preserve');
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

options = zeros(1, 6);
options(1) = 1e-8;     % convergence criterion (relative change in fit)
options(2) = 2;        % random orthogonalised initial values
options(3) = 0;        % no plots
options(5) = NaN;      % no progress output
options(6) = maxIt;    % maximum iterations
const = [2 2 2];       % non-negativity on every mode

best = struct('err', Inf);
errs = nan(nStart, 1); its = nan(nStart, 1);
t0 = tic;
for k = 1:nStart
    rng(k);
    [F, it, err, cc] = parafac(Xn, rank, options, const);
    errs(k) = err; its(k) = it;
    if err < best.err
        best = struct('F', {F}, 'err', err, 'it', it, 'cc', cc);
    end
end
seconds = toc(t0);

A = best.F{1}; B = best.F{2}; C = best.F{3};
nb = vecnorm(B); nc = vecnorm(C);
B = B ./ nb; C = C ./ nc; A = A .* (nb .* nc);
scores = A .* scale(:);                       % back to R.U.
fmax = scores .* max(B, [], 1) .* max(C, [], 1);

model = zeros(size(Xn));
for f = 1:rank
    model = model + A(:, f) .* reshape(B(:, f), 1, []) .* reshape(C(:, f), 1, 1, []);
end
obs = ~isnan(Xn);
rsq = 1 - sum((Xn(obs) - model(obs)).^2) / sum(Xn(obs).^2);
sameMin = sum(errs <= min(errs) * (1 + 1e-3));

comp = "C" + string(1:rank);
writetable(array2table([ex(:) B], 'VariableNames', ["ex" comp]), ...
           fullfile(outDir, 'pf_ex_loadings.csv'));
writetable(array2table([em(:) C], 'VariableNames', ["em" comp]), ...
           fullfile(outDir, 'pf_em_loadings.csv'));
writetable([table(samples, 'VariableNames', "sample") array2table(fmax, 'VariableNames', comp)], ...
           fullfile(outDir, 'pf_fmax.csv'));
v = ver('MATLAB');
info = table(rank, nStart, maxIt, 1e-8, rsq, best.it, sameMin / nStart, best.cc, seconds, ...
             string(v.Version), 'VariableNames', ["rank" "nstart" "maxit" "ctol" "rsq" ...
             "iter" "converged_fraction" "corcondia" "seconds" "version"]);
info.engine = "N-way toolbox parafac (MATLAB " + string(v.Version) + ")";
writetable(info, fullfile(outDir, 'pf_info.csv'));
disp(info);
end
