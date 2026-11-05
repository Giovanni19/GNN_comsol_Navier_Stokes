function results = run_gnn_vs_standard_transitions( ...
    model, G, t_comsol, dt_comsol, k_values, bdf_order, ...
    initial_guess)

% ============================================================
% RUN GNN VS STANDARD TRANSITIONS
%
% For every transition k:
%
%       X_k -> X_(k+1)
%
% solve the stationary BDF-equivalent problem twice:
%
%   1. STANDARD:
%          X^(0) = X_k
%
%   2. SECOND INITIAL GUESS:
%          GNN / velocity-only / pressure-only / standard
%
% The two solves use exactly the same stationary equations.
% Only the nonlinear initial guess changes.
% ============================================================

import com.comsol.model.util.*

k_values = k_values(:);
num_test_steps = length(k_values);

%% ============================================================
% VALIDATE REQUESTED TRANSITIONS
% ============================================================

num_transitions = length(t_comsol) - 1;

if any(k_values < 1) || any(k_values > num_transitions)
    error( ...
        ['Requested COMSOL transitions are outside the available ' ...
         'trajectory.\nAvailable range: 1 -> %d\n' ...
         'Requested range: %d -> %d'], ...
        num_transitions, ...
        min(k_values), ...
        max(k_values));
end

if ~isfield(G, 'comsol_transition_index')
    error( ...
        ['G does not contain comsol_transition_index. ' ...
         'prepare_gnn_comsol_pipeline.m must create the mapping.']);
end

available_gnn_transitions = G.comsol_transition_index(:);

missing_transitions = setdiff( ...
    k_values, ...
    available_gnn_transitions);

if ~isempty(missing_transitions)
    error( ...
        ['No GNN predictions are available for the requested ' ...
         'COMSOL transition(s): %s\n' ...
         'Available GNN transitions: %d -> %d'], ...
        mat2str(missing_transitions.'), ...
        min(available_gnn_transitions), ...
        max(available_gnn_transitions));
end

%% Defaults

if nargin < 6 || isempty(bdf_order)
    bdf_order = 1;
end

if nargin < 7 || isempty(initial_guess)
    cfg = gnn_comsol_config();
    initial_guess = cfg.initial_guess;
end

valid_guesses = { ...
    'gnn', ...
    'gnn_velocity_only', ...
    'gnn_pressure_only', ...
    'standard', ...
    'standard_via_init'};

if ~ismember(initial_guess, valid_guesses)
    error( ...
        ['Unknown initial_guess %s. Expected one of: ' ...
         'gnn, gnn_velocity_only, gnn_pressure_only, ' ...
         'standard, standard_via_init.'], ...
        initial_guess);
end

if ~ismember(bdf_order, [1, 2])
    error('bdf_order must be either 1 or 2.');
end

if bdf_order == 2 && any(k_values < 2)
    error( ...
        ['BDF2 requires two previous states. ' ...
         'All transition indices must satisfy k >= 2.']);
end

fprintf( ...
    '\nInitial guess: %s | BDF%d | transitions: %d\n', ...
    initial_guess, bdf_order, num_test_steps);

%% ============================================================
% 1. SET UP STATIONARY SOLVER
% ============================================================



spf = model.component('comp1').physics('spf');

%% Inlet boundary condition

inlet_feature = spf.feature('inl1');

inlet_bc_type = char( ...
    inlet_feature.getString('BoundaryCondition'));

switch inlet_bc_type

    case 'FullyDevelopedFlow'
        inlet_property = 'Uavfdf';

    case 'Velocity'
        inlet_property = 'U0in';

    otherwise
        error( ...
            'Unsupported inlet BoundaryCondition: %s', ...
            inlet_bc_type);
end

% Store original time-dependent expression
inlet_time_expression = char( ...
    inlet_feature.getString(inlet_property));

if ~contains(inlet_time_expression, 't[1/s]')
    warning( ...
        ['The inlet expression does not contain t[1/s]. ' ...
         'No explicit time substitution will be performed.']);
end

%% Physics / solver features

init1 = spf.feature('init1');
vf    = spf.feature('vf1');

% Dependent Variables node
v1 = model.sol('sol3').feature('v1');




%% ============================================================
% 2. GNN INITIAL-GUESS SETUP
% ============================================================

x_gnn = G.node_coordinates(:,1);
y_gnn = G.node_coordinates(:,2);

% Coordinates in mphinterp format
P_eval = G.node_coordinates.';

cfg = gnn_comsol_config();

u_gnn_file = cfg.u_gnn_file;
v_gnn_file = cfg.v_gnn_file;
p_gnn_file = cfg.p_gnn_file;


%% ============================================================
% 3. PREALLOCATE RESULTS
% ============================================================

standard_time = nan(num_test_steps,1);
gnn_time      = nan(num_test_steps,1);

standard_iterations = nan(num_test_steps,1);
gnn_iterations      = nan(num_test_steps,1);

% Nonlinear solver statistics
standard_res_evals = nan(num_test_steps,1);
gnn_res_evals      = nan(num_test_steps,1);

standard_jac_evals = nan(num_test_steps,1);
gnn_jac_evals      = nan(num_test_steps,1);

standard_linear_solves = nan(num_test_steps,1);
gnn_linear_solves      = nan(num_test_steps,1);

standard_converged = false(num_test_steps,1);
gnn_converged      = false(num_test_steps,1);

% Standard stationary vs GNN stationary
rmse_standard_gnn_u = nan(num_test_steps,1);
rmse_standard_gnn_v = nan(num_test_steps,1);
rmse_standard_gnn_p = nan(num_test_steps,1);

% Standard stationary vs TD
rmse_standard_td_u = nan(num_test_steps,1);
rmse_standard_td_v = nan(num_test_steps,1);
rmse_standard_td_p = nan(num_test_steps,1);

% GNN stationary vs TD
rmse_gnn_td_u = nan(num_test_steps,1);
rmse_gnn_td_v = nan(num_test_steps,1);
rmse_gnn_td_p = nan(num_test_steps,1);

% Spatial error fields
error_standard_td_u = cell(num_test_steps,1);
error_standard_td_v = cell(num_test_steps,1);
error_standard_td_p = cell(num_test_steps,1);

error_gnn_td_u = cell(num_test_steps,1);
error_gnn_td_v = cell(num_test_steps,1);
error_gnn_td_p = cell(num_test_steps,1);

% Time-discretization information
dt_previous = nan(num_test_steps,1);
dt_current  = nan(num_test_steps,1);
step_ratio  = nan(num_test_steps,1);

bdf_a0 = nan(num_test_steps,1);
bdf_a1 = nan(num_test_steps,1);
bdf_a2 = nan(num_test_steps,1);


%% ============================================================
% 4. LOOP OVER TRANSITIONS
% ============================================================

for i = 1:num_test_steps

    k = k_values(i);

    %% --------------------------------------------------------
    % Current transition
    %
    %       X_k -> X_(k+1)
    % ---------------------------------------------------------

    t_prev   = t_comsol(k);
    t_target = t_comsol(k+1);

    dt = t_target - t_prev;
    dt_current(i) = dt;

    if bdf_order == 2

        dt_previous(i) = ...
            t_prev - t_comsol(k-1);

        step_ratio(i) = ...
            dt_current(i) / dt_previous(i);
    end


    %% ========================================================
    % TIME-DEPENDENT TARGET
    % ========================================================

    u_td = mphinterp( ...
        model, ...
        'u', ...
        'coord', P_eval, ...
        'dataset', 'dset4', ...
        'solnum', k+1);

    v_td = mphinterp( ...
        model, ...
        'v', ...
        'coord', P_eval, ...
        'dataset', 'dset4', ...
        'solnum', k+1);

    p_td = mphinterp( ...
        model, ...
        'p', ...
        'coord', P_eval, ...
        'dataset', 'dset4', ...
        'solnum', k+1);

    u_td = u_td(:);
    v_td = v_td(:);
    p_td = p_td(:);

    fprintf('\n----------------------------------------\n');
    fprintf( ...
        'Transition %d/%d | k = %d\n', ...
        i, num_test_steps, k);


    %% ========================================================
    % 4.1 UPDATE TIME STEP
    % ========================================================

    dt_str = sprintf('%.17g[s]', dt);
    model.param.set('dt_step', dt_str);


    %% ========================================================
    % 4.2 UPDATE TIME-DEPENDENT INLET BC
    % ========================================================

    t_target_str = sprintf('%.17g', t_target);

    inlet_target_expr = strrep( ...
        inlet_time_expression, ...
        't[1/s]', ...
        t_target_str);

    inlet_feature.set( ...
        inlet_property, ...
        inlet_target_expr);



    %% ========================================================
    % 4.3 UPDATE BDF-EQUIVALENT VOLUME FORCE
    % ========================================================

    % Current state X_k
    u_prev_expr = sprintf( ...
        "withsol('sol4',u,setind(t,%d))", ...
        k);

    v_prev_expr = sprintf( ...
        "withsol('sol4',v,setind(t,%d))", ...
        k);

    if bdf_order == 1

        % BDF1:
        %
        % du/dt = (u_(k+1) - u_k)/dt
        %
        % Equivalent volume force:
        %
        % Fx = rho*(u_k-u)/dt
        % Fy = rho*(v_k-v)/dt

        bdf_a0(i) = 1;
        bdf_a1(i) = -1;
        bdf_a2(i) = 0;

        Fx = sprintf( ...
            'spf.rho*(%s-u)/dt_step', ...
            u_prev_expr);

        Fy = sprintf( ...
            'spf.rho*(%s-v)/dt_step', ...
            v_prev_expr);

    else

        % Variable-step BDF2

        h_n   = dt_current(i);
        h_nm1 = dt_previous(i);

        if h_n <= 0 || h_nm1 <= 0
            error( ...
                'Non-positive timestep detected at k=%d.', ...
                k);
        end

        w_n = h_n / h_nm1;

        a0 = (1 + 2*w_n) / (1 + w_n);
        a1 = -(1 + w_n);
        a2 = w_n^2 / (1 + w_n);

        bdf_a0(i) = a0;
        bdf_a1(i) = a1;
        bdf_a2(i) = a2;

        u_prevprev_expr = sprintf( ...
            "withsol('sol4',u,setind(t,%d))", ...
            k-1);

        v_prevprev_expr = sprintf( ...
            "withsol('sol4',v,setind(t,%d))", ...
            k-1);

        Fx = sprintf( ...
            ['-spf.rho*(%.17g*u + %.17g*(%s) ' ...
             '+ %.17g*(%s))/dt_step'], ...
            a0, ...
            a1, u_prev_expr, ...
            a2, u_prevprev_expr);

        Fy = sprintf( ...
            ['-spf.rho*(%.17g*v + %.17g*(%s) ' ...
             '+ %.17g*(%s))/dt_step'], ...
            a0, ...
            a1, v_prev_expr, ...
            a2, v_prevprev_expr);
    end

    vf.set('F', {Fx; Fy; '0'});


    %% ========================================================
    % 4.4 CONFIGURE STANDARD INITIAL GUESS
    % ========================================================

    % Reset Initial Values
    init1.setIndex('u_init', '0', 0);
    init1.setIndex('u_init', '0', 1);
    init1.setIndex('u_init', '0', 2);
    init1.set('p_init', '0');

    % Standard initial guess = X_k from time-dependent solution
    v1.set('initmethod', 'sol');
    v1.set('initsol', 'sol4');
    v1.set('solnum', num2str(k));


    %% ========================================================
    % 4.5 RUN STANDARD STATIONARY SOLVER
    % ========================================================

    standard_log = fullfile( ...
        tempdir, ...
        sprintf('comsol_standard_k%d.log', k));

    if isfile(standard_log)
        delete(standard_log);
    end

    ModelUtil.showProgress(standard_log);

    tic;

    try

        model.sol('sol3').runAll();

        standard_time(i) = toc;
        standard_converged(i) = true;

        ModelUtil.showProgress(false);

        [standard_iterations(i), standard_info] = ...
            get_nonlinear_iterations(standard_log);

        fprintf('\nSTANDARD NONLINEAR HISTORY:\n');
        disp(standard_info.header);
        disp(standard_info.table_data);

        standard_res_evals(i) = ...
            standard_info.total_residual_evaluations;

        standard_jac_evals(i) = ...
            standard_info.total_jacobian_evaluations;

        standard_linear_solves(i) = ...
            standard_info.total_linear_solves;

    catch ME

        standard_time(i) = toc;
        standard_converged(i) = false;

        ModelUtil.showProgress(false);

        fprintf( ...
            'FAILED | %.6f s\n', ...
            standard_time(i));

        fprintf( ...
            'Error at k = %d:\n%s\n', ...
            k, ME.message);
    end


    %% ========================================================
    % 4.6 STORE STANDARD SOLUTION
    % ========================================================

    if standard_converged(i)

        u_standard = mphinterp( ...
            model, ...
            'u', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        v_standard = mphinterp( ...
            model, ...
            'v', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        p_standard = mphinterp( ...
            model, ...
            'p', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        u_standard = u_standard(:);
        v_standard = v_standard(:);
        p_standard = p_standard(:);

        % Spatial errors
        error_standard_td_u{i} = ...
            abs(u_standard - u_td);

        error_standard_td_v{i} = ...
            abs(v_standard - v_td);

        error_standard_td_p{i} = ...
            abs(p_standard - p_td);

        % RMSE
        rmse_standard_td_u(i) = sqrt( ...
            mean((u_standard - u_td).^2));

        rmse_standard_td_v(i) = sqrt( ...
            mean((v_standard - v_td).^2));

        rmse_standard_td_p(i) = sqrt( ...
            mean((p_standard - p_td).^2));
    end


    %% ========================================================
    % 4.7 GET GNN PREDICTION
    % ========================================================

    matching_rows = find( ...
        G.comsol_transition_index(:) == k);
    
    if numel(matching_rows) ~= 1
        error( ...
            'COMSOL transition %d maps to %d GNN prediction rows.', ...
            k, numel(matching_rows));
    end
    
    gnn_row = matching_rows(1);
    % Sanity check against physical target time
    target_time_tolerance = 1e-6;

    time_mismatch = abs( ...
        G.target_time(gnn_row) - t_target);

    if time_mismatch > target_time_tolerance
        error( ...
            ['Internal GNN/COMSOL alignment inconsistency.\n' ...
             'COMSOL transition = %d\n' ...
             'GNN row           = %d\n' ...
             'COMSOL target     = %.15e s\n' ...
             'GNN target        = %.15e s\n' ...
             'Difference        = %.15e s'], ...
            k, ...
            gnn_row, ...
            t_target, ...
            G.target_time(gnn_row), ...
            time_mismatch);
    end

    u_gnn_pred = G.u_pred(gnn_row,:).';
    v_gnn_pred = G.v_pred(gnn_row,:).';
    p_gnn_pred = G.p_pred(gnn_row,:).';


    %% ========================================================
    % 4.8 UPDATE GNN INTERPOLATION FILES
    % ========================================================
    writematrix( ...
        [x_gnn, y_gnn, u_gnn_pred], ...
        u_gnn_file, ...
        'Delimiter', 'space');

    writematrix( ...
        [x_gnn, y_gnn, v_gnn_pred], ...
        v_gnn_file, ...
        'Delimiter', 'space');

    writematrix( ...
        [x_gnn, y_gnn, p_gnn_pred], ...
        p_gnn_file, ...
        'Delimiter', 'space');

   
%% ========================================================
% 4.8b IMPORT GNN FIELDS INTO COMSOL
% ========================================================

% --------------------------------------------------------
% Remove previous GNN interpolation functions, if present
% --------------------------------------------------------

gnn_tags = {'gnnU', 'gnnV', 'gnnP'};

existing_tags = cell(model.func.tags());

for q = 1:numel(gnn_tags)

    if any(strcmp(existing_tags, gnn_tags{q}))
        model.func.remove(gnn_tags{q});
    end

end


% --------------------------------------------------------
% U interpolation
% --------------------------------------------------------

model.func.create('gnnU', 'Interpolation');

fU = model.func('gnnU');

fU.label('GNN velocity u');

fU.set('source', 'file');
fU.set('filename', u_gnn_file);
fU.set('struct', 'spreadsheet');

% First import lets COMSOL inspect the three columns
fU.refresh();

% Column 1 = x
% Column 2 = y
% Column 3 = u
fU.set('columnType', { ...
    'col1', 'arg', ...
    'col2', 'arg', ...
    'col3', 'value'});

fU.set('funcnames', { ...
    'col1', 'uGNN', ...
    'col2', 'uGNN', ...
    'col3', 'uGNN'});

fU.set('funcnametable', {'uGNN', '1'});

fU.set('argunit', {'m', 'm'});
fU.set('fununit', {'m/s'});

fU.set('interp', 'linear');
fU.set('extrap', 'const');

fU.refresh();


% --------------------------------------------------------
% V interpolation
% --------------------------------------------------------

model.func.create('gnnV', 'Interpolation');

fV = model.func('gnnV');

fV.label('GNN velocity v');

fV.set('source', 'file');
fV.set('filename', v_gnn_file);
fV.set('struct', 'spreadsheet');

fV.refresh();

fV.set('columnType', { ...
    'col1', 'arg', ...
    'col2', 'arg', ...
    'col3', 'value'});

fV.set('funcnames', { ...
    'col1', 'vGNN', ...
    'col2', 'vGNN', ...
    'col3', 'vGNN'});

fV.set('funcnametable', {'vGNN', '1'});

fV.set('argunit', {'m', 'm'});
fV.set('fununit', {'m/s'});

fV.set('interp', 'linear');
fV.set('extrap', 'const');

fV.refresh();


% --------------------------------------------------------
% P interpolation
% --------------------------------------------------------

model.func.create('gnnP', 'Interpolation');

fP = model.func('gnnP');

fP.label('GNN pressure');

fP.set('source', 'file');
fP.set('filename', p_gnn_file);
fP.set('struct', 'spreadsheet');

fP.refresh();

fP.set('columnType', { ...
    'col1', 'arg', ...
    'col2', 'arg', ...
    'col3', 'value'});

fP.set('funcnames', { ...
    'col1', 'pGNN', ...
    'col2', 'pGNN', ...
    'col3', 'pGNN'});

fP.set('funcnametable', {'pGNN', '1'});

fP.set('argunit', {'m', 'm'});
fP.set('fununit', {'Pa'});

fP.set('interp', 'linear');
fP.set('extrap', 'const');

fP.refresh();

    %% ========================================================
    % 4.9 CONFIGURE SECOND INITIAL GUESS
    % ========================================================

    p_prev_expr = sprintf( ...
        "withsol('sol4',p,setind(t,%d))", ...
        k);

    switch initial_guess

        case 'gnn'
        
            % u,v,p from GNN
            init1.setIndex('u_init', 'uGNN(x,y)', 0);
            init1.setIndex('u_init', 'vGNN(x,y)', 1);
            init1.setIndex('u_init', '0',         2);
            init1.set('p_init', 'pGNN(x,y)');
        
            v1.set('initmethod', 'init');


        case 'gnn_velocity_only'

            % Velocity from GNN, pressure from X_k
            init1.setIndex('u_init', 'uGNN(x,y)', 0);
            init1.setIndex('u_init', 'vGNN(x,y)', 1);
            init1.setIndex('u_init', '0',         2);
        
            init1.set( ...
                'p_init', ...
                char(p_prev_expr));
        
            v1.set('initmethod', 'init');


        case 'gnn_pressure_only'

            % Velocity from X_k, pressure from GNN
            init1.setIndex( ...
                'u_init', ...
                char(u_prev_expr), ...
                0);

            init1.setIndex( ...
                'u_init', ...
                char(v_prev_expr), ...
                1);

            init1.setIndex( ...
                'u_init', ...
                '0', ...
                2);
            init1.set( ...
                'p_init', ...
                'pGNN(x,y)');

            v1.set('initmethod', 'init');


        case 'standard_via_init'

            % Same X_k as STANDARD, but passed through
            % the physics Initial Values node.

            init1.setIndex( ...
                'u_init', ...
                char(u_prev_expr), ...
                0);

            init1.setIndex( ...
                'u_init', ...
                char(v_prev_expr), ...
                1);

            init1.setIndex( ...
                'u_init', ...
                '0', ...
                2);

            init1.set( ...
                'p_init', ...
                char(p_prev_expr));

            v1.set('initmethod', 'init');


        case 'standard'

            % X_k directly from sol4
            init1.setIndex('u_init', '0', 0);
            init1.setIndex('u_init', '0', 1);
            init1.setIndex('u_init', '0', 2);
            init1.set('p_init', '0');

            v1.set('initmethod', 'sol');
            v1.set('initsol', 'sol4');
            v1.set('solnum', num2str(k));
    end


    %% ========================================================
    % 4.10 RUN SECOND STATIONARY SOLVER
    % ========================================================

    gnn_log = fullfile( ...
        tempdir, ...
        sprintf('comsol_gnn_k%d.log', k));

    if isfile(gnn_log)
        delete(gnn_log);
    end

    ModelUtil.showProgress(gnn_log);

    tic;

    try

        model.sol('sol3').runAll();

        gnn_time(i) = toc;
        gnn_converged(i) = true;

        ModelUtil.showProgress(false);

        [gnn_iterations(i), gnn_info] = ...
            get_nonlinear_iterations(gnn_log);

        fprintf('\nGNN NONLINEAR HISTORY:\n');
        disp(gnn_info.header);
        disp(gnn_info.table_data);
        gnn_res_evals(i) = ...
            gnn_info.total_residual_evaluations;

        gnn_jac_evals(i) = ...
            gnn_info.total_jacobian_evaluations;

        gnn_linear_solves(i) = ...
            gnn_info.total_linear_solves;

    catch ME

        gnn_time(i) = toc;
        gnn_converged(i) = false;

        ModelUtil.showProgress(false);

        fprintf( ...
            'FAILED | %.6f s\n', ...
            gnn_time(i));

        fprintf( ...
            'Error at k = %d (%s):\n%s\n', ...
            k, initial_guess, ME.message);
    end


    %% ========================================================
    % 4.11 STORE SECOND SOLUTION
    % ========================================================

    if gnn_converged(i)

        u_gnn = mphinterp( ...
            model, ...
            'u', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        v_gnn = mphinterp( ...
            model, ...
            'v', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        p_gnn = mphinterp( ...
            model, ...
            'p', ...
            'coord', P_eval, ...
            'dataset', 'dset3');

        u_gnn = u_gnn(:);
        v_gnn = v_gnn(:);
        p_gnn = p_gnn(:);

        error_gnn_td_u{i} = abs(u_gnn - u_td);
        error_gnn_td_v{i} = abs(v_gnn - v_td);
        error_gnn_td_p{i} = abs(p_gnn - p_td);

        rmse_gnn_td_u(i) = sqrt( ...
            mean((u_gnn - u_td).^2));

        rmse_gnn_td_v(i) = sqrt( ...
            mean((v_gnn - v_td).^2));

        rmse_gnn_td_p(i) = sqrt( ...
            mean((p_gnn - p_td).^2));
    end


    %% ========================================================
    % 4.12 COMPARE FINAL STATIONARY SOLUTIONS
    % ========================================================

    if standard_converged(i) && gnn_converged(i)

        rmse_standard_gnn_u(i) = sqrt( ...
            mean((u_gnn - u_standard).^2));

        rmse_standard_gnn_v(i) = sqrt( ...
            mean((v_gnn - v_standard).^2));

        rmse_standard_gnn_p(i) = sqrt( ...
            mean((p_gnn - p_standard).^2));
    end


%% ========================================================
% 4.13 TRANSITION SUMMARY
% ========================================================

if standard_converged(i) && gnn_converged(i)

    iterations_saved = ...
        standard_iterations(i) - gnn_iterations(i);

    iteration_reduction = ...
        100 * iterations_saved / standard_iterations(i);

    fprintf( ...
        ['k=%d | Newton %d -> %d | saved %+d (%+.1f%%) | ' ...
         'time %.3f -> %.3f s\n'], ...
        k, ...
        standard_iterations(i), ...
        gnn_iterations(i), ...
        iterations_saved, ...
        iteration_reduction, ...
        standard_time(i), ...
        gnn_time(i));

    fprintf( ...
        ['      work: Res %d->%d | Jac %d->%d | ' ...
         'Sol %d->%d\n'], ...
        standard_res_evals(i), ...
        gnn_res_evals(i), ...
        standard_jac_evals(i), ...
        gnn_jac_evals(i), ...
        standard_linear_solves(i), ...
        gnn_linear_solves(i));

    fprintf( ...
        ['      RMSE STD-GNN: ' ...
         'u=%.3e | v=%.3e | p=%.3e\n'], ...
        rmse_standard_gnn_u(i), ...
        rmse_standard_gnn_v(i), ...
        rmse_standard_gnn_p(i));

else

    fprintf( ...
        'k=%d | STD converged=%d | %s converged=%d\n', ...
        k, ...
        standard_converged(i), ...
        initial_guess, ...
        gnn_converged(i));
end

end
%% ============================================================
% 5. RESTORE MODEL
% ============================================================

fprintf('\n========================================\n');
fprintf('RESTORING MODEL\n');
fprintf('========================================\n');

% Restore physics initial values
init1.setIndex('u_init', '0', 0);
init1.setIndex('u_init', '0', 1);
init1.setIndex('u_init', '0', 2);
init1.set('p_init', '0');

% Restore original inlet expression
inlet_feature.set( ...
    inlet_property, ...
    inlet_time_expression);

fprintf('Initial velocity restored to [0, 0, 0].\n');
fprintf('Initial pressure restored to 0.\n');
fprintf('Original inlet expression restored:\n');
fprintf('  %s\n', inlet_time_expression);


%% ============================================================
% 6. FINAL SUMMARY
% ============================================================

fprintf('\n========================================\n');
fprintf('FINAL SUMMARY\n');
fprintf('========================================\n');

fprintf( ...
    'Transitions tested: %d\n\n', ...
    num_test_steps);

fprintf( ...
    'Standard converged: %d / %d\n', ...
    sum(standard_converged), ...
    num_test_steps);

fprintf( ...
    'Second guess converged: %d / %d\n', ...
    sum(gnn_converged), ...
    num_test_steps);

if any(standard_converged)
    mean_standard_time = ...
        mean(standard_time(standard_converged));
else
    mean_standard_time = NaN;
end

if any(gnn_converged)
    mean_gnn_time = ...
        mean(gnn_time(gnn_converged));
else
    mean_gnn_time = NaN;
end

fprintf( ...
    '\nMean standard time = %.6f s\n', ...
    mean_standard_time);

fprintf( ...
    'Mean GNN time      = %.6f s\n', ...
    mean_gnn_time);

both_converged = ...
    standard_converged & gnn_converged;

if any(both_converged)

    paired_standard_time = ...
        standard_time(both_converged);

    paired_gnn_time = ...
        gnn_time(both_converged);

    total_standard_time = ...
        sum(paired_standard_time);

    total_gnn_time = ...
        sum(paired_gnn_time);

    total_speedup = ...
        total_standard_time / total_gnn_time;

    fprintf( ...
        '\nBoth converged: %d / %d\n', ...
        sum(both_converged), ...
        num_test_steps);

    fprintf( ...
        'Paired standard total = %.6f s\n', ...
        total_standard_time);

    fprintf( ...
        'Paired GNN total      = %.6f s\n', ...
        total_gnn_time);

    fprintf( ...
        'Paired speedup        = %.6fx\n', ...
        total_speedup);

else

    fprintf( ...
        '\nNo transitions converged with both methods.\n');
end


%% ============================================================
% 7. BUILD RESULTS STRUCT
% ============================================================

results.initial_guess = initial_guess;
results.bdf_order     = bdf_order;

results.k        = k_values;
results.t_prev   = t_comsol(k_values);
results.t_target = t_comsol(k_values + 1);
results.dt       = dt_current;

results.standard_iterations = standard_iterations;
results.gnn_iterations      = gnn_iterations;

% Nonlinear solver work
results.standard_res_evals = standard_res_evals;
results.gnn_res_evals      = gnn_res_evals;

results.standard_jac_evals = standard_jac_evals;
results.gnn_jac_evals      = gnn_jac_evals;

results.standard_linear_solves = standard_linear_solves;
results.gnn_linear_solves      = gnn_linear_solves;

results.iterations_saved = ...
    standard_iterations - gnn_iterations;

results.iteration_reduction_percent = ...
    100 * ...
    (standard_iterations - gnn_iterations) ...
    ./ standard_iterations;

results.standard_converged = standard_converged;
results.gnn_converged      = gnn_converged;

% RMSE stationary standard vs stationary GNN
results.rmse_standard_gnn_u = rmse_standard_gnn_u;
results.rmse_standard_gnn_v = rmse_standard_gnn_v;
results.rmse_standard_gnn_p = rmse_standard_gnn_p;

% RMSE stationary standard vs TD
results.rmse_standard_td_u = rmse_standard_td_u;
results.rmse_standard_td_v = rmse_standard_td_v;
results.rmse_standard_td_p = rmse_standard_td_p;

% RMSE stationary GNN vs TD
results.rmse_gnn_td_u = rmse_gnn_td_u;
results.rmse_gnn_td_v = rmse_gnn_td_v;
results.rmse_gnn_td_p = rmse_gnn_td_p;

% Spatial error fields
results.node_coordinates = G.node_coordinates;

results.error_standard_td_u = error_standard_td_u;
results.error_standard_td_v = error_standard_td_v;
results.error_standard_td_p = error_standard_td_p;

results.error_gnn_td_u = error_gnn_td_u;
results.error_gnn_td_v = error_gnn_td_v;
results.error_gnn_td_p = error_gnn_td_p;

% BDF information
results.dt_current  = dt_current;
results.dt_previous = dt_previous;
results.step_ratio  = step_ratio;

results.bdf_a0 = bdf_a0;
results.bdf_a1 = bdf_a1;
results.bdf_a2 = bdf_a2;


%% ============================================================
% 8. BUILD SOLVER COMPARISON TABLE
% ============================================================

solver_table = table( ...
    results.k, ...
    results.t_prev, ...
    results.t_target, ...
    results.dt, ...
    results.standard_converged, ...
    results.gnn_converged, ...
    results.standard_iterations, ...
    results.gnn_iterations, ...
    results.standard_res_evals, ...
    results.gnn_res_evals, ...
    results.standard_jac_evals, ...
    results.gnn_jac_evals, ...
    results.standard_linear_solves, ...
    results.gnn_linear_solves, ...
    'VariableNames', { ...
        'k', ...
        't_prev', ...
        't_target', ...
        'dt', ...
        'STD_converged', ...
        'GNN_converged', ...
        'STD_iterations', ...
        'GNN_iterations', ...
        'STD_res_evals', ...
        'GNN_res_evals', ...
        'STD_jac_evals', ...
        'GNN_jac_evals', ...
        'STD_linear_solves', ...
        'GNN_linear_solves' ...
    });


%% BDF information

solver_table.BDF_order = ...
    repmat(bdf_order, height(solver_table), 1);

solver_table.dt_previous = dt_previous;
solver_table.dt_current  = dt_current;
solver_table.step_ratio  = step_ratio;

solver_table.BDF_a0 = bdf_a0;
solver_table.BDF_a1 = bdf_a1;
solver_table.BDF_a2 = bdf_a2;


%% Solver savings

solver_table.iterations_saved = ...
    solver_table.STD_iterations - ...
    solver_table.GNN_iterations;

solver_table.jac_evals_saved = ...
    solver_table.STD_jac_evals - ...
    solver_table.GNN_jac_evals;

solver_table.linear_solves_saved = ...
    solver_table.STD_linear_solves - ...
    solver_table.GNN_linear_solves;

solver_table.iteration_reduction_percent = ...
    100 * ...
    solver_table.iterations_saved ./ ...
    solver_table.STD_iterations;

solver_table.jac_reduction_percent = ...
    100 * ...
    solver_table.jac_evals_saved ./ ...
    solver_table.STD_jac_evals;

solver_table.linear_solves_reduction_percent = ...
    100 * ...
    solver_table.linear_solves_saved ./ ...
    solver_table.STD_linear_solves;

results.solver_table = solver_table;


%% ============================================================
% 9. SAVE SOLVER COMPARISON TABLE
% ============================================================

excel_file = 'solver_comparison.xlsx';

writetable( ...
    solver_table, ...
    excel_file);

fprintf( ...
    '\nSolver comparison table saved to:\n%s\n', ...
    excel_file);


%% ============================================================
% 10. PLOT SOLVER COMPARISON
% ============================================================

plot_nonlinear_iterations(solver_table);

end