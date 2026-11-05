function [model, G, t_comsol, dt_comsol, ...
          num_snapshots, num_transitions, ...
          num_predictions, num_nodes] = ...
          prepare_gnn_comsol_pipeline()

% ============================================================
% PREPARE GNN -> COMSOL PIPELINE
%
% This function performs the common initialization required by
% the GNN-COMSOL stationary experiments:
%
%   1. Load the COMSOL model containing the computed sol1
%   2. Read the time-dependent solution times
%   3. Run Python GNN inference
%   4. Load the GNN predictions
%   5. Check the prediction arrays
%   6. Check COMSOL <-> GNN timestep alignment
%
% Outputs
% -------
% model
%     Loaded COMSOL model.
%
% G
%     Structure containing the GNN predictions.
%
% t_comsol
%     Stored COMSOL solution times.
%
% dt_comsol
%     Time-step duration for each transition.
%
% num_snapshots
%     Number of stored COMSOL snapshots.
%
% num_transitions
%     Number of COMSOL transitions.
%
% num_predictions
%     Number of GNN predictions.
%
% num_nodes
%     Number of GNN nodes.
%
% ============================================================

import com.comsol.model.*
import com.comsol.model.util.*


%% ============================================================
% 1. FILE PATHS
% ============================================================

% All paths are centralized in gnn_comsol_config.m so that every
% pipeline script stays in sync when the model, GNN environment,
% or trained run directory change.
cfg = gnn_comsol_config();

model_file           = cfg.model_file;
dataset_file         = cfg.dataset_file;
python_exe           = cfg.python_exe;
python_script        = cfg.python_script;
run_dir              = cfg.run_dir;
gnn_predictions_file = cfg.gnn_predictions_file;


%% ============================================================
% 2. LOAD COMSOL MODEL
% ============================================================

fprintf('\n========================================\n');
fprintf('LOADING COMSOL MODEL\n');
fprintf('========================================\n');

model = mphload(model_file);

fprintf('COMSOL model loaded successfully.\n');
fprintf('Model:\n%s\n', model_file);


%% ============================================================
% 3. CHECK TIME-DEPENDENT SOLUTION SOL1
% ============================================================

fprintf('\n========================================\n');
fprintf('CHECKING TIME-DEPENDENT SOLUTION\n');
fprintf('========================================\n');

% Read information specifically from sol1
sol1_info = mphsolinfo(model, 'soltag', 'sol4');

% Stored time values
t_comsol = sol1_info.solvals(:);

% ------------------------------------------------------------
% Remove consecutive duplicate COMSOL solution times
% ------------------------------------------------------------
%
% COMSOL can occasionally store more than one solution at the
% same physical time. These are not distinct physical
% transitions and must not be counted twice.

time_tol = 100 * eps(max(1, max(abs(t_comsol))));

duplicate_time = [false; abs(diff(t_comsol)) <= time_tol];

if any(duplicate_time)

    fprintf('\nWARNING: duplicate COMSOL solution times detected:\n');

    duplicate_indices = find(duplicate_time);

    for q = 1:length(duplicate_indices)

        idx = duplicate_indices(q);

        fprintf( ...
            '  snapshot %d: t = %.17g duplicates snapshot %d\n', ...
            idx, ...
            t_comsol(idx), ...
            idx - 1);

    end

    t_comsol(duplicate_time) = [];

end

num_snapshots = length(t_comsol);
num_transitions = num_snapshots - 1;

fprintf('Number of COMSOL snapshots   = %d\n', ...
    num_snapshots);

fprintf('Number of COMSOL transitions = %d\n', ...
    num_transitions);

fprintf('Initial time = %.15e s\n', ...
    t_comsol(1));

fprintf('Final time   = %.15e s\n', ...
    t_comsol(end));

% Time step associated with every transition
dt_comsol = diff(t_comsol);


%% ============================================================
% 4. RUN PYTHON GNN INFERENCE
% ============================================================

fprintf('\n========================================\n');
fprintf('RUNNING GNN INFERENCE\n');
fprintf('========================================\n');

% The predictions file is passed explicitly: evaluate_test.py used to
% write it to a path hard-coded in its own source, which had to agree
% with cfg.gnn_predictions_file by hand. Now this config is the only
% place where that location is written down.
cmd = sprintf( ...
    ['"%s" "%s" "%s" --dataset "%s" ' ...
     '--predictions-mat "%s" --no-animation 2>&1'], ...
    python_exe, ...
    python_script, ...
    run_dir, ...
    dataset_file, ...
    gnn_predictions_file);

fprintf('Python command:\n%s\n\n', cmd);

tic;

[status, python_output] = system(cmd);

gnn_elapsed_time = toc;

% Print Python output in MATLAB
fprintf('%s\n', python_output);

if status ~= 0

    fprintf('\n========================================\n');
    fprintf('GNN INFERENCE FAILED\n');
    fprintf('========================================\n');

    fprintf('\nPython exit status: %d\n', status);
    fprintf('\nPython output:\n');
    fprintf('%s\n', python_output);

    error( ...
        'Python script returned status %d.\nPython output:\n%s', ...
        status, python_output);

end

fprintf('\n========================================\n');
fprintf('GNN INFERENCE COMPLETED\n');
fprintf('========================================\n');

fprintf('Elapsed time = %.6f s\n', ...
    gnn_elapsed_time);


%% ============================================================
% 5. LOAD GNN PREDICTIONS
% ============================================================

fprintf('\n========================================\n');
fprintf('LOADING GNN PREDICTIONS\n');
fprintf('========================================\n');

if ~isfile(gnn_predictions_file)

    error( ...
        'GNN prediction file not found:\n%s', ...
        gnn_predictions_file);

end

G = load(gnn_predictions_file);

num_predictions = size(G.u_pred, 1);
num_nodes = size(G.u_pred, 2);

fprintf('Number of GNN predictions = %d\n', ...
    num_predictions);

fprintf('Number of GNN nodes       = %d\n', ...
    num_nodes);


%% ============================================================
% 6. BASIC GNN ARRAY CHECKS
% ============================================================

if size(G.v_pred, 1) ~= num_predictions || ...
   size(G.p_pred, 1) ~= num_predictions

    error( ...
        ['u_pred, v_pred and p_pred have inconsistent ' ...
         'numbers of samples.']);

end


if size(G.v_pred, 2) ~= num_nodes || ...
   size(G.p_pred, 2) ~= num_nodes

    error( ...
        ['u_pred, v_pred and p_pred have inconsistent ' ...
         'numbers of nodes.']);

end


dt_gnn = G.delta_t(:);

if length(dt_gnn) ~= num_predictions

    error( ...
        ['Number of GNN delta_t values does not match ' ...
         'predictions.']);

end


%% ============================================================
% 7. CHECK COMSOL <-> GNN ALIGNMENT
% ============================================================

fprintf('\n========================================\n');
fprintf('CHECKING COMSOL <-> GNN ALIGNMENT\n');
fprintf('========================================\n');

% ============================================================
% The prediction file defines which transitions were evaluated
% ============================================================
%
% IMPORTANT:
%
% run_config.dataset.skip_initial describes TRAINING.
% It must NOT be used to infer which transitions are present in
% gnn_predictions.mat.
%
% evaluate_test.py may evaluate:
%   - the complete trajectory,
%   - only the training range,
%   - a held-out subset,
%   - or another selected range.
%
% target_time written by Python is therefore the source of truth.

if ~isfield(G, 'target_time')
    error( ...
        'GNN prediction file does not contain target_time.');
end

target_time_gnn = G.target_time(:);

if length(target_time_gnn) ~= num_predictions
    error( ...
        ['Number of GNN target_time values does not match ' ...
         'the number of predictions.']);
end

% ============================================================
% Match every GNN prediction to a COMSOL transition
% ============================================================

gnn_transition_indices = zeros(num_predictions, 1);

target_time_tolerance = 1e-6;

for k = 1:num_predictions

    % Transition j ends at COMSOL snapshot j+1.
    differences = abs( ...
        t_comsol(2:end) - target_time_gnn(k));

    [minimum_difference, transition_index] = min(differences);

    if minimum_difference > target_time_tolerance

        error( ...
            ['Could not match GNN prediction %d to a COMSOL ' ...
             'transition.\n' ...
             'GNN target time = %.15e s\n' ...
             'Closest COMSOL target time = %.15e s\n' ...
             'Difference = %.15e s'], ...
            k, ...
            target_time_gnn(k), ...
            t_comsol(transition_index + 1), ...
            minimum_difference);
    end

    gnn_transition_indices(k) = transition_index;
end

% There must not be two predictions for the same transition.
if length(unique(gnn_transition_indices)) ~= num_predictions
    error( ...
        'Multiple GNN predictions map to the same COMSOL transition.');
end

% ============================================================
% Compare delta_t for every matched transition
% ============================================================

dt_comsol_gnn = dt_comsol(gnn_transition_indices);

if length(dt_comsol_gnn) ~= length(dt_gnn)
    error( ...
        ['Internal alignment error.\n' ...
         'Matched COMSOL dt = %d\n' ...
         'GNN dt            = %d'], ...
        length(dt_comsol_gnn), ...
        length(dt_gnn));
end

dt_difference = abs(dt_comsol_gnn - dt_gnn);

max_dt_error = max(dt_difference);

dt_abs_tolerance = 1e-6;
dt_rel_tolerance = 1e-4;

dt_tolerance = max( ...
    dt_abs_tolerance, ...
    dt_rel_tolerance * max(abs(dt_comsol_gnn)) ...
);

fprintf( ...
    'Maximum |dt_COMSOL - dt_GNN| = %.15e s\n', ...
    max_dt_error);

fprintf( ...
    'delta_t alignment tolerance  = %.15e s\n', ...
    dt_tolerance);

if max_dt_error > dt_tolerance

    error( ...
        ['COMSOL and GNN delta_t arrays are not aligned.\n' ...
         'Maximum difference = %.15e s\n' ...
         'Tolerance          = %.15e s'], ...
        max_dt_error, ...
        dt_tolerance);
end

% ============================================================
% Compare target times
% ============================================================

target_time_comsol = ...
    t_comsol(gnn_transition_indices + 1);

target_time_difference = abs( ...
    target_time_comsol - target_time_gnn);

max_target_time_error = max(target_time_difference);

fprintf( ...
    'Maximum |target_time_COMSOL - target_time_GNN| = %.15e s\n', ...
    max_target_time_error);

if max_target_time_error > target_time_tolerance

    error( ...
        ['COMSOL and GNN target times are not aligned.\n' ...
         'Maximum difference = %.15e s\n' ...
         'Tolerance          = %.15e s'], ...
        max_target_time_error, ...
        target_time_tolerance);
end

% ============================================================
% Report matched range
% ============================================================

first_comsol_transition = min(gnn_transition_indices);
last_comsol_transition  = max(gnn_transition_indices);

fprintf('First matched COMSOL transition = %d\n', ...
    first_comsol_transition);

fprintf('Last matched COMSOL transition  = %d\n', ...
    last_comsol_transition);

fprintf('Number of GNN predictions       = %d\n', ...
    num_predictions);

fprintf('\nCOMSOL and GNN are correctly aligned.\n');

G.comsol_transition_index = gnn_transition_indices;
%% ============================================================
% 8. SHOW FIRST TRANSITIONS
% ============================================================

fprintf('\n========================================\n');
fprintf('FIRST TRANSITIONS\n');
fprintf('========================================\n');

num_to_print = min(5, num_predictions);

for k = 1:num_to_print

    transition_index = gnn_transition_indices(k);

    fprintf( ...
        ['prediction = %d | transition = %d | ' ...
         't_prev = %.8e | ' ...
         't_target = %.8e | ' ...
         'dt = %.8e\n'], ...
        k, ...
        transition_index, ...
        t_comsol(transition_index), ...
        t_comsol(transition_index + 1), ...
        dt_gnn(k));

end


fprintf('\n========================================\n');
fprintf('INITIAL PIPELINE READY\n');
fprintf('========================================\n');

end