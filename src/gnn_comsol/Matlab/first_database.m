%% ============================================================
% COMSOL -> GNN DATASET
% ============================================================

clc
close all

import com.comsol.model.*
import com.comsol.model.util.*


%% ============================================================
% 1. CONFIGURATION
% ============================================================

cfg = get_config();


%% ============================================================
% 2. LOAD COMSOL MODEL
% ============================================================

model = mphload(cfg.model_file);


%% ============================================================
% 3. RUN TIME-DEPENDENT STUDY
% ============================================================

run_time_dependent_study(model, cfg);


%% ============================================================
% 4. MESH
% ============================================================

[P, T, meshdata] = extract_mesh(model);

N  = size(P, 2);
Ne = size(T, 2);


%% ============================================================
% 5. TIME INFORMATION
% ============================================================

t = extract_time_vector(model, cfg.solution_tag);

Nt = length(t);


%% ============================================================
% 6. SOLUTION FIELDS
% ============================================================

X = extract_solution_fields(model, P, cfg);

[Nt_X, N_X, ~] = size(X);

if Nt_X ~= Nt || N_X ~= N
    error('Unexpected dimensions for X.');
end
%% ============================================================
% 7. PHYSICS FEATURES
% ============================================================

[physics_features, physics_feature_names] = ...
    extract_physics_features(model, P, cfg);

if size(physics_features, 1) ~= Nt || ...
   size(physics_features, 2) ~= N

    error('Unexpected dimensions for physics_features.');
end

%% ============================================================
% 8. GEOMETRY FEATURES
% ============================================================

[geometry_features, geometry_feature_names] = ...
    extract_geometry_features(model, P, cfg);

if size(geometry_features, 1) ~= N
    error('Unexpected dimensions for geometry_features.');
end

%% ============================================================
% 9. BOUNDARY CONDITIONS
% ============================================================

bc = extract_boundary_conditions( ...
    model, meshdata, P, t, cfg);

%% ============================================================
% 10. REFERENCE VELOCITY
% ============================================================

U_mean = mphevaluate(model, 'U_mean', 'm/s');

if ~isscalar(U_mean) || ~isfinite(U_mean) || U_mean <= 0
    error('Invalid COMSOL parameter U_mean.');
end

U_inf = U_mean;

fprintf( ...
    'Normalization: U_mean = %.6g m/s | U_inf = %.6g m/s\n', ...
    U_mean, U_inf);
%% ============================================================
% 11. GRAPH
% ============================================================

graph = build_graph(P, T);

%% ============================================================
% 12. WLSQ OPERATORS
% ============================================================

[G_wlsq, neighbors_python] = ...
    build_wlsq_operators(P, graph.edges);

%% ============================================================
% 13. FLUID PROPERTIES
% ============================================================

fluid = extract_fluid_properties(model, P, cfg);

%% ============================================================
% 14. FEM JACOBIANS
% ============================================================

if cfg.export_jacobian

    jacobian = extract_fem_jacobians( ...
        model, P, t, cfg);

else

    jacobian = [];

end

%% ============================================================
% 15. VALIDATE DATASET
% ============================================================

validate_dataset( ...
    X, ...
    physics_features, ...
    geometry_features, ...
    bc, ...
    graph, ...
    jacobian, ...
    t, ...
    cfg);

%% ============================================================
% 16. SAVE DATASET
% ============================================================

save_dataset( ...
    cfg, ...
    X, ...
    physics_features, physics_feature_names, ...
    geometry_features, geometry_feature_names, ...
    bc, graph, ...
    T, P, t, ...
    G_wlsq, neighbors_python, ...
    U_inf, fluid, jacobian);

%% ============================================================
% 17. SAVE COMSOL MODEL
% ============================================================

if cfg.save_comsol_model

    fprintf('\nSaving COMSOL model...\n');

    mphsave(model, cfg.model_file);

    fprintf( ...
        'COMSOL model saved successfully:\n%s\n', ...
        cfg.model_file);

end
%% ============================================================
% LOCAL FUNCTIONS
% ============================================================

function cfg = get_config()
    %% COMSOL model
    cfg.model_file = ...
        '"\\nl-filer1\users$\giovanni\Desktop\Comsol simulations\Reynolds dataset\variable step size\channel2d_manual_stabilization_100R_cylinder_stretch_free_timestep.mph"';

    %% Execution
    cfg.run_simulation = false;
    cfg.save_comsol_model = false;

    %% COMSOL solution
    cfg.solution_tag = 'sol4';
    cfg.dataset_tag = 'dset4';

    %% Geometry dataset
    cfg.geometry_dataset_tag = 'dset2';

    %% Solution variables
    cfg.u_var = 'u';
    cfg.v_var = 'v';
    cfg.p_var = 'p';

    %% Physics-derived variables
    cfg.du_dx_var = 'du_dx';
    cfg.du_dy_var = 'du_dy';
    cfg.dv_dx_var = 'dv_dx';
    cfg.dv_dy_var = 'dv_dy';
    cfg.div_conv_var = 'div_conv';

    %% Jacobian export
    cfg.export_jacobian = false;

    cfg.jacobian_timestep_variable_tag = ...
        'var_jacobian_export';

    cfg.jacobian_matrix_type = 'K';
    cfg.jacobian_rowscale = false;
    cfg.jacobian_linearization = 'target_state';

    %% Output dataset
    cfg.output_file = ...
        'C:\Users\giovanni\.comsol\v64\llmatlab\channel2d_manual_stabilization_100R_cylinder_stretch_free_timestep.mat';
end

function run_time_dependent_study(model, cfg)

    if cfg.run_simulation

        fprintf('\n========================================\n');
        fprintf('RUNNING STUDY 1 - TIME DEPENDENT\n');
        fprintf('========================================\n');

        tic;

        try

            model.study('std1').run;

            elapsed_time = toc;

            fprintf('Elapsed time = %.6f s\n', elapsed_time);

        catch ME

            elapsed_time = toc;

            fprintf('\n========================================\n');
            fprintf('STUDY 1 FAILED\n');
            fprintf('========================================\n');

            fprintf('Elapsed time = %.6f s\n\n', elapsed_time);
            fprintf('%s\n', ME.message);

            rethrow(ME);

        end

    else

        fprintf('\n========================================\n');
        fprintf('SKIPPING STUDY 1\n');
        fprintf('Using solution already stored in COMSOL model.\n');
        fprintf('========================================\n');

    end

end

function [P, T, meshdata] = extract_mesh(model)

    [~, meshdata] = mphmeshstats(model, "mesh1");

    % Mesh vertex coordinates
    P = meshdata.vertex;

    % Find triangular elements
    tri_idx = find(strcmp(meshdata.types, 'tri'), 1);

    if isempty(tri_idx)
        error('No triangular elements found in mesh.');
    end

    % Triangle connectivity
    T = meshdata.elem{tri_idx};

    % Convert COMSOL zero-based indexing to MATLAB indexing
    if min(T(:)) == 0
        T = T + 1;
    end

end


function t = extract_time_vector(model, solution_tag)

    info = mphsolinfo(model, 'soltag', solution_tag);

    t = info.solvals;

end

function X = extract_solution_fields(model, P, cfg)

    u_nodes = mphinterp( ...
        model, cfg.u_var, ...
        'coord', P, ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all');

    v_nodes = mphinterp( ...
        model, cfg.v_var, ...
        'coord', P, ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all');

    p_nodes = mphinterp( ...
        model, cfg.p_var, ...
        'coord', P, ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all');

    X = cat(3, ...
        u_nodes, ...
        v_nodes, ...
        p_nodes);

end

function [physics_features, feature_names] = ...
    extract_physics_features(model, P, cfg)

    du_dx = mphinterp( ...
        model, cfg.du_dx_var, ...
        'coord', P, ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all');

    du_dy = mphinterp( ...
        model, cfg.du_dy_var, ...
        'coord', P, ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all');

    dv_dx = mphinterp( ...
        model, cfg.dv_dx_var, ...
        'coord', P, ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all');

    dv_dy = mphinterp( ...
        model, cfg.dv_dy_var, ...
        'coord', P, ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all');

    div_conv = mphinterp( ...
        model, cfg.div_conv_var, ...
        'coord', P, ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all');

    physics_features = cat(3, ...
        du_dx, ...
        du_dy, ...
        dv_dx, ...
        dv_dy, ...
        div_conv);

    feature_names = {
        'du_dx', ...
        'du_dy', ...
        'dv_dx', ...
        'dv_dy', ...
        'div_conv'
    };

end

function [geometry_features, feature_names] = ...
    extract_geometry_features(model, P, cfg)

    % Wall
    G_wall = mphinterp( ...
        model, 'G2', ...
        'coord', P, ...
        'dataset', cfg.geometry_dataset_tag);

    wall_dir_x = mphinterp( ...
        model, 'wd.Ddirx', ...
        'coord', P, ...
        'dataset', cfg.geometry_dataset_tag);

    wall_dir_y = mphinterp( ...
        model, 'wd.Ddiry', ...
        'coord', P, ...
        'dataset', cfg.geometry_dataset_tag);

    % Inlet
    G_inlet = mphinterp( ...
        model, 'G3', ...
        'coord', P, ...
        'dataset', cfg.geometry_dataset_tag);

    inlet_dir_x = mphinterp( ...
        model, 'wd2.Ddirx', ...
        'coord', P, ...
        'dataset', cfg.geometry_dataset_tag);

    inlet_dir_y = mphinterp( ...
        model, 'wd2.Ddiry', ...
        'coord', P, ...
        'dataset', cfg.geometry_dataset_tag);

    % Outlet
    G_outlet = mphinterp( ...
        model, 'G4', ...
        'coord', P, ...
        'dataset', cfg.geometry_dataset_tag);

    outlet_dir_x = mphinterp( ...
        model, 'wd3.Ddirx', ...
        'coord', P, ...
        'dataset', cfg.geometry_dataset_tag);

    outlet_dir_y = mphinterp( ...
        model, 'wd3.Ddiry', ...
        'coord', P, ...
        'dataset', cfg.geometry_dataset_tag);

    % Build geometric features
    wall_geom_x = G_wall .* wall_dir_x;
    wall_geom_y = G_wall .* wall_dir_y;

    inlet_geom_x = G_inlet .* inlet_dir_x;
    inlet_geom_y = G_inlet .* inlet_dir_y;

    outlet_geom_x = G_outlet .* outlet_dir_x;
    outlet_geom_y = G_outlet .* outlet_dir_y;

    geometry_features = [
        wall_geom_x(:), ...
        wall_geom_y(:), ...
        inlet_geom_x(:), ...
        inlet_geom_y(:), ...
        outlet_geom_x(:), ...
        outlet_geom_y(:)
    ];

    feature_names = {
        'wall_geom_x', ...
        'wall_geom_y', ...
        'inlet_geom_x', ...
        'inlet_geom_y', ...
        'outlet_geom_x', ...
        'outlet_geom_y'
    };

end

function bc = extract_boundary_conditions( ...
    model, meshdata, P, t, cfg)

    N  = size(P, 2);
    Nt = length(t);

    spf = model.component('comp1').physics('spf');


    %% --------------------------------------------------------
    % Boundary mesh
    % ---------------------------------------------------------

    edg_idx = find(strcmp(meshdata.types, 'edg'), 1);

    if isempty(edg_idx)
        error('No boundary edge elements found in COMSOL mesh.');
    end

    boundary_edges = meshdata.elem{edg_idx};

    boundary_entity = meshdata.elementity{edg_idx};
    boundary_entity = boundary_entity(:);

    if min(boundary_edges(:)) == 0
        boundary_edges = boundary_edges + 1;
    end


    %% --------------------------------------------------------
    % Allocate masks and prescribed values
    % ---------------------------------------------------------

    wall_mask   = false(N, 1);
    inlet_mask  = false(N, 1);
    outlet_mask = false(N, 1);

    velocity_dirichlet_mask = false(N, 1);
    pressure_dirichlet_mask = false(N, 1);

    velocity_bc = zeros(Nt, N, 2);
    pressure_bc = zeros(Nt, N);


    %% --------------------------------------------------------
    % Walls
    % ---------------------------------------------------------

    wall_feature = spf.feature('wallbc1');

    wall_boundaries = ...
        wall_feature.selection.entities(1);

    wall_edge_mask = ...
        ismember(boundary_entity, wall_boundaries);

    wall_nodes = ...
        unique(boundary_edges(:, wall_edge_mask));

    wall_nodes = wall_nodes(:);

    if isempty(wall_nodes)
        error('No wall nodes found.');
    end

    wall_mask(wall_nodes) = true;

    velocity_dirichlet_mask(wall_nodes) = true;

    velocity_bc(:, wall_nodes, 1) = 0;
    velocity_bc(:, wall_nodes, 2) = 0;


    %% --------------------------------------------------------
    % Inlet
    % ---------------------------------------------------------

    inlet_feature = spf.feature('inl1');

    inlet_boundaries = ...
        inlet_feature.selection.entities(1);

    inlet_edge_mask = ...
        ismember(boundary_entity, inlet_boundaries);

    inlet_nodes = ...
        unique(boundary_edges(:, inlet_edge_mask));

    inlet_nodes = inlet_nodes(:);

    if isempty(inlet_nodes)
        error('No inlet nodes found.');
    end

    inlet_mask(inlet_nodes) = true;

    velocity_dirichlet_mask(inlet_nodes) = true;


    %% --------------------------------------------------------
    % Prescribed inlet velocity expression
    % ---------------------------------------------------------

    U0_expr = ...
        char(inlet_feature.getString('U0in'));

    U0 = mphinterp( ...
        model, ...
        U0_expr, ...
        'coord', P(:, inlet_nodes), ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all', ...
        'edim', 'boundary');


    %% --------------------------------------------------------
    % Inlet boundary normal
    % ---------------------------------------------------------

    nx_inlet = mphinterp( ...
        model, ...
        'nx', ...
        'coord', P(:, inlet_nodes), ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all', ...
        'edim', 'boundary');

    ny_inlet = mphinterp( ...
        model, ...
        'ny', ...
        'coord', P(:, inlet_nodes), ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all', ...
        'edim', 'boundary');


    %% --------------------------------------------------------
    % Expand static quantities over time if necessary
    % ---------------------------------------------------------

    if size(U0, 1) == 1 && Nt > 1
        U0 = repmat(U0, Nt, 1);
    end

    if size(nx_inlet, 1) == 1 && Nt > 1
        nx_inlet = repmat(nx_inlet, Nt, 1);
    end

    if size(ny_inlet, 1) == 1 && Nt > 1
        ny_inlet = repmat(ny_inlet, Nt, 1);
    end


    %% --------------------------------------------------------
    % Prescribed inlet velocity
    % ---------------------------------------------------------

    velocity_bc(:, inlet_nodes, 1) = ...
        -U0 .* nx_inlet;

    velocity_bc(:, inlet_nodes, 2) = ...
        -U0 .* ny_inlet;


    %% --------------------------------------------------------
    % Outlet
    % ---------------------------------------------------------

    outlet_feature = spf.feature('out1');

    outlet_boundaries = ...
        outlet_feature.selection.entities(1);

    outlet_edge_mask = ...
        ismember(boundary_entity, outlet_boundaries);

    outlet_nodes = ...
        unique(boundary_edges(:, outlet_edge_mask));

    outlet_nodes = outlet_nodes(:);

    if isempty(outlet_nodes)
        error('No outlet nodes found.');
    end

    outlet_mask(outlet_nodes) = true;

    pressure_dirichlet_mask(outlet_nodes) = true;


    %% --------------------------------------------------------
    % Prescribed outlet pressure
    % ---------------------------------------------------------

    p0_expr = ...
        char(outlet_feature.getString('p0'));

    p0 = mphinterp( ...
        model, ...
        p0_expr, ...
        'coord', P(:, outlet_nodes), ...
        'dataset', cfg.dataset_tag, ...
        'solnum', 'all', ...
        'edim', 'boundary');

    if size(p0, 1) == 1 && Nt > 1
        p0 = repmat(p0, Nt, 1);
    end

    pressure_bc(:, outlet_nodes) = p0;


    %% --------------------------------------------------------
    % Corner convention
    % ---------------------------------------------------------

    shared_wall_inlet_nodes = ...
        wall_mask & inlet_mask;

    velocity_bc(:, shared_wall_inlet_nodes, 1) = 0;
    velocity_bc(:, shared_wall_inlet_nodes, 2) = 0;


    %% --------------------------------------------------------
    % Diagnostics
    % ---------------------------------------------------------

    fprintf('\n========================================\n');
    fprintf('BOUNDARY CONDITIONS\n');
    fprintf('========================================\n');

    fprintf('Wall nodes               : %d\n', ...
        sum(wall_mask));

    fprintf('Inlet nodes              : %d\n', ...
        sum(inlet_mask));

    fprintf('Outlet nodes             : %d\n', ...
        sum(outlet_mask));

    fprintf('Velocity Dirichlet nodes : %d\n', ...
        sum(velocity_dirichlet_mask));

    fprintf('Pressure Dirichlet nodes : %d\n', ...
        sum(pressure_dirichlet_mask));

    fprintf('Wall/inlet shared nodes  : %d\n', ...
        sum(shared_wall_inlet_nodes));


    %% --------------------------------------------------------
    % Output structure
    % ---------------------------------------------------------

    bc.wall_mask = wall_mask;
    bc.inlet_mask = inlet_mask;
    bc.outlet_mask = outlet_mask;

    bc.velocity_dirichlet_mask = ...
        velocity_dirichlet_mask;

    bc.pressure_dirichlet_mask = ...
        pressure_dirichlet_mask;

    bc.velocity_bc = velocity_bc;
    bc.pressure_bc = pressure_bc;

    bc.shared_wall_inlet_nodes = ...
        shared_wall_inlet_nodes;

end

function graph = build_graph(P, T)

    N  = size(P, 2);
    Ne = size(T, 2);


    %% --------------------------------------------------------
    % Global mesh size
    % ---------------------------------------------------------

    h_elem = zeros(Ne, 1);

    for e = 1:Ne

        nodes = T(:, e);

        p1 = P(:, nodes(1));
        p2 = P(:, nodes(2));
        p3 = P(:, nodes(3));

        d12 = norm(p1 - p2);
        d23 = norm(p2 - p3);
        d31 = norm(p3 - p1);

        h_elem(e) = max([d12, d23, d31]);

    end

    h = max(h_elem);


    %% --------------------------------------------------------
    % Unique undirected mesh edges
    % ---------------------------------------------------------

    edges = zeros(3 * Ne, 2);

    k = 1;

    for e = 1:Ne

        nodes = T(:, e);

        edges(k, :)     = [nodes(1), nodes(2)];
        edges(k + 1, :) = [nodes(2), nodes(3)];
        edges(k + 2, :) = [nodes(3), nodes(1)];

        k = k + 3;

    end

    edges = sort(edges, 2);
    edges = unique(edges, 'rows');

    Nedges = size(edges, 1);


    %% --------------------------------------------------------
    % Edge geometry
    % ---------------------------------------------------------

    i = edges(:, 1);
    j = edges(:, 2);

    dx = P(1, j) - P(1, i);
    dy = P(2, j) - P(2, i);

    dx = dx(:);
    dy = dy(:);

    dij = sqrt(dx.^2 + dy.^2);


    %% --------------------------------------------------------
    % GCN edge weights
    % ---------------------------------------------------------

    weights = 1 ./ (1 + dij / h);

    weights = weights(:);


    %% --------------------------------------------------------
    % PyTorch Geometric directed graph
    % ---------------------------------------------------------

    source = [i; j];
    target = [j; i];

    % Zero-based indexing for Python
    edge_index = [source - 1, target - 1];

    % Same weight in both directions
    edge_weight = [weights; weights];

    % Triangle connectivity in Python indexing
    cell_index = T.' - 1;


    %% --------------------------------------------------------
    % Output
    % ---------------------------------------------------------

    graph.edges = edges;
    graph.Nedges = Nedges;

    graph.h = h;

    graph.edge_index = edge_index;
    graph.edge_weight = edge_weight;

    graph.cell_index = cell_index;

end

function [G_wlsq, neighbors_python] = ...
    build_wlsq_operators(P, edges)

    N = size(P, 2);
    Nedges = size(edges, 1);

    fprintf('\n========================================\n');
    fprintf('WLSQ GEOMETRIC PRECOMPUTATION\n');
    fprintf('========================================\n');


    %% --------------------------------------------------------
    % Neighbor list
    % ---------------------------------------------------------

    neighbors = cell(N, 1);

    for e = 1:Nedges

        node_i = edges(e, 1);
        node_j = edges(e, 2);

        neighbors{node_i}(end + 1) = node_j;
        neighbors{node_j}(end + 1) = node_i;

    end


    %% --------------------------------------------------------
    % WLSQ operators
    % ---------------------------------------------------------

    G_wlsq = cell(N, 1);

    for node_i = 1:N

        neigh = neighbors{node_i};
        neigh = neigh(:);

        xi = P(1, node_i);
        yi = P(2, node_i);

        xj = P(1, neigh).';
        yj = P(2, neigh).';

        dx_i = xj - xi;
        dy_i = yj - yi;

        r_i = sqrt(dx_i.^2 + dy_i.^2);

        if any(r_i <= eps)
            error( ...
                'Zero or invalid neighbor distance found at node %d.', ...
                node_i);
        end

        w_i = 1 ./ r_i;

        A_i = [
            w_i .* dx_i, ...
            w_i .* dy_i
        ];

        if rank(A_i) < 2
            error( ...
                'WLSQ matrix is rank deficient at node %d.', ...
                node_i);
        end

        [Q_i, R_i] = qr(A_i, 0);

        M_i = R_i \ Q_i.';

        G_wlsq{node_i} = M_i .* w_i.';

    end


    %% --------------------------------------------------------
    % Python zero-based neighbor indexing
    % ---------------------------------------------------------

    neighbors_python = cell(N, 1);

    for node_i = 1:N
        neighbors_python{node_i} = ...
            neighbors{node_i} - 1;
    end

end

function fluid = extract_fluid_properties(model, P, cfg)

    fluid.export = true;

    try

        rho_nodes = mphinterp( ...
            model, 'spf.rho', ...
            'coord', P, ...
            'dataset', cfg.dataset_tag, ...
            'solnum', 'all');

        mu_nodes = mphinterp( ...
            model, 'spf.mu', ...
            'coord', P, ...
            'dataset', cfg.dataset_tag, ...
            'solnum', 'all');

    catch ME

        fluid.export = false;

        fprintf( ...
            'Could not evaluate spf.rho / spf.mu:\n  %s\n', ...
            ME.message);

        fprintf([ ...
            'The dataset will be written WITHOUT rho and mu. ', ...
            'Put them in the experiment YAML, or fix the ', ...
            'variable names above.\n']);

        return

    end


    %% --------------------------------------------------------
    % Check that properties are constant
    % ---------------------------------------------------------

    rho_spread = ...
        max(rho_nodes(:)) - min(rho_nodes(:));

    mu_spread = ...
        max(mu_nodes(:)) - min(mu_nodes(:));

    rho = mean(rho_nodes(:));
    mu  = mean(mu_nodes(:));

    if rho_spread > 1e-9 * abs(rho) || ...
       mu_spread > 1e-9 * abs(mu)

        fluid.export = false;

        fprintf([ ...
            '\nFluid properties are NOT constant over the mesh ', ...
            'or over time.\n' ...
            'The export of rho and mu is skipped.\n']);

        return

    end


    %% --------------------------------------------------------
    % Output
    % ---------------------------------------------------------

    fluid.rho = rho;
    fluid.mu = mu;

end

function jacobian = extract_fem_jacobians(model, P, t, cfg)

    N = size(P, 2);
    Nt = length(t);
    num_transitions = Nt - 1;

    solution_tag = cfg.solution_tag;
    var_tag = cfg.jacobian_timestep_variable_tag;


    %% ========================================================
    % VERIFY SOURCE TIME-DEPENDENT SOLUTION
    % ========================================================

    source_info_before = ...
        mphsolinfo(model, 'soltag', solution_tag);

    num_source_solutions_before = ...
        length(source_info_before.solvals);

    if num_source_solutions_before ~= Nt

        error( ...
            ['Unexpected number of TD solutions before Jacobian ' ...
             'extraction: %d instead of %d.'], ...
            num_source_solutions_before, Nt);

    end


    %% ========================================================
    % CREATE TEMPORARY TIMESTEP VARIABLE
    % ========================================================

    try
        model.component('comp1').variable.remove(var_tag);
    catch
    end

    model.component('comp1').variable.create(var_tag);


    %% ========================================================
    % CREATE DEDICATED JACOBIAN ASSEMBLY SOLVER
    % ========================================================

    jacobian_solver_tag = 'solJ';
    jacobian_study_step_tag = 'stJ';
    jacobian_variables_tag = 'vJ';

    try
        model.sol.remove(jacobian_solver_tag);
    catch
    end

    model.sol.create(jacobian_solver_tag);

    model.sol(jacobian_solver_tag).study('std1');


    % Study step
    model.sol(jacobian_solver_tag).create( ...
        jacobian_study_step_tag, ...
        'StudyStep');

    model.sol(jacobian_solver_tag) ...
        .feature(jacobian_study_step_tag) ...
        .set('study', 'std1');

    model.sol(jacobian_solver_tag) ...
        .feature(jacobian_study_step_tag) ...
        .set('studystep', 'time');


    % Dependent variables
    model.sol(jacobian_solver_tag).create( ...
        jacobian_variables_tag, ...
        'Variables');

    model.sol(jacobian_solver_tag) ...
        .feature(jacobian_variables_tag) ...
        .set('control', 'time');


    fprintf('\n========================================\n');
    fprintf('FEM JACOBIAN EXTRACTION\n');
    fprintf('========================================\n');

    fprintf('Source TD solver     : %s\n', solution_tag);
    fprintf('Assembly solver      : %s\n', jacobian_solver_tag);
    fprintf('TD snapshots present : %d\n', Nt);


    %% ========================================================
    % ALLOCATE STORAGE
    % ========================================================

    jacobian_sparse = cell(num_transitions, 1);

    jacobian_shape = [];

    jacobian_perm = [];


    %% ========================================================
    % ASSEMBLE J_k = K(X_(k+1))
    % ========================================================

    try

        for k = 1:num_transitions

            target_solnum = k + 1;

            dt_k = t(k+1) - t(k);

            if ~isfinite(dt_k) || dt_k <= 0

                error( ...
                    'Invalid timestep at transition %d: %.17e', ...
                    k, dt_k);

            end


            %% ------------------------------------------------
            % Supply timestep for COMSOL stabilization
            % -------------------------------------------------

            timestep_expression = ...
                sprintf('%.17g[s]', dt_k);

            model.component('comp1') ...
                .variable(var_tag) ...
                .set( ...
                    'timestep', ...
                    timestep_expression);


            %% ------------------------------------------------
            % Assemble Jacobian at target state
            % -------------------------------------------------

            fprintf( ...
                ['Jacobian %d/%d | transition %d | ' ...
                 'solnum %d | t = %.8e s | ' ...
                 'dt = %.8e s ... '], ...
                k, ...
                num_transitions, ...
                k, ...
                target_solnum, ...
                t(target_solnum), ...
                dt_k);


            S = mphmatrix( ...
                model, ...
                jacobian_solver_tag, ...
                'out', {'K'}, ...
                'initmethod', 'sol', ...
                'initsol', solution_tag, ...
                'solnum', target_solnum, ...
                'extractafter', jacobian_variables_tag, ...
                'rowscale', 'off');


            J_k = S.K;


            %% =================================================
            % BUILD COMSOL -> GNN PERMUTATION
            %
            % This only needs to be done once because the DOF
            % ordering does not change between timesteps.
            % =================================================

            if k == 1

                xdJ = mphxmeshinfo( ...
                    model, ...
                    'soltag', jacobian_solver_tag, ...
                    'studysteptag', jacobian_variables_tag);

                names = xdJ.dofs.dofnames;
                nameinds = xdJ.dofs.nameinds;
                solvectorinds = xdJ.dofs.solvectorinds;
                dof_coords = xdJ.dofs.coords;


                %% --------------------------------------------
                % Dimension consistency
                % ---------------------------------------------

                if xdJ.ndofs ~= 3*N

                    error( ...
                        ['Variables-node xmesh has %d DOFs, ' ...
                         'but expected 3*N = %d.'], ...
                        xdJ.ndofs, ...
                        3*N);

                end

                if size(J_k,1) ~= xdJ.ndofs || ...
                   size(J_k,2) ~= xdJ.ndofs

                    error( ...
                        ['Jacobian size %d x %d does not match ' ...
                         'Variables-node xmesh size %d.'], ...
                        size(J_k,1), ...
                        size(J_k,2), ...
                        xdJ.ndofs);

                end


                %% --------------------------------------------
                % Identify p/u/v DOFs
                % ---------------------------------------------

                id_p = find(strcmp(names, 'comp1.p')) - 1;
                id_u = find(strcmp(names, 'comp1.u')) - 1;
                id_v = find(strcmp(names, 'comp1.v')) - 1;

                if numel(id_p) ~= 1 || ...
                   numel(id_u) ~= 1 || ...
                   numel(id_v) ~= 1

                    error( ...
                        'Could not uniquely identify p/u/v DOF names.');

                end

                valid_nameinds = [id_p, id_u, id_v];

                if any(~ismember( ...
                        double(nameinds(:)), ...
                        valid_nameinds))

                    error( ...
                        ['Variables-node xmesh contains DOFs ' ...
                         'other than p, u, v.']);

                end


                %% --------------------------------------------
                % Sort DOFs by COMSOL solution-vector position
                % ---------------------------------------------

                [~, order] = sort(solvectorinds);

                nameinds_sorted = nameinds(order);
                dof_coords_sorted = dof_coords(:, order);


                %% --------------------------------------------
                % Match COMSOL nodes to GNN mesh nodes
                % ---------------------------------------------

                comsol_to_gnn = zeros(N, 1);

                max_puv_coord_error = 0;
                max_match_error = 0;

                for j = 1:N

                    q = 3*(j-1) + 1;


                    % Expected ordering: p, u, v
                    if ~strcmp( ...
                            names{nameinds_sorted(q)+1}, ...
                            'comp1.p') || ...
                       ~strcmp( ...
                            names{nameinds_sorted(q+1)+1}, ...
                            'comp1.u') || ...
                       ~strcmp( ...
                            names{nameinds_sorted(q+2)+1}, ...
                            'comp1.v')

                        error( ...
                            ['Unexpected p/u/v ordering at COMSOL ' ...
                             'triplet %d.'], ...
                            j);

                    end


                    cp = dof_coords_sorted(:, q);
                    cu = dof_coords_sorted(:, q+1);
                    cv = dof_coords_sorted(:, q+2);


                    max_puv_coord_error = max( ...
                        [max_puv_coord_error, ...
                         norm(cp-cu), ...
                         norm(cp-cv)]);


                    dist = vecnorm(P - cp, 2, 1);

                    [dmin, gnn_node] = min(dist);

                    comsol_to_gnn(j) = gnn_node;

                    max_match_error = max( ...
                        max_match_error, ...
                        dmin);

                end


                %% --------------------------------------------
                % Validate physical correspondence
                % ---------------------------------------------

                if max_puv_coord_error > 1e-12

                    error( ...
                        ['p/u/v DOFs do not share the same ' ...
                         'coordinates. Maximum error = %.3e.'], ...
                        max_puv_coord_error);

                end

                if max_match_error > 1e-12

                    error( ...
                        ['COMSOL/GNN coordinate matching failed. ' ...
                         'Maximum error = %.3e.'], ...
                        max_match_error);

                end

                if numel(unique(comsol_to_gnn)) ~= N

                    error( ...
                        ['COMSOL -> GNN node mapping is not ' ...
                         'one-to-one.']);

                end


                %% --------------------------------------------
                % Invert node mapping
                % ---------------------------------------------

                gnn_to_comsol = zeros(N, 1);

                for j = 1:N

                    gnn_to_comsol( ...
                        comsol_to_gnn(j)) = j;

                end


                %% --------------------------------------------
                % Expand node mapping to DOF permutation
                % ---------------------------------------------

                jacobian_perm = zeros(3*N, 1);

                for i = 1:N

                    j = gnn_to_comsol(i);

                    jacobian_perm(3*i-2) = 3*j-2;
                    jacobian_perm(3*i-1) = 3*j-1;
                    jacobian_perm(3*i)   = 3*j;

                end


                %% --------------------------------------------
                % Verify permutation
                % ---------------------------------------------

                if ~isequal( ...
                        sort(jacobian_perm), ...
                        (1:3*N).')

                    error( ...
                        'Jacobian DOF permutation is invalid.');

                end


                %% --------------------------------------------
                % Diagnostics
                % ---------------------------------------------

                num_reordered_nodes = sum( ...
                    comsol_to_gnn ~= (1:N).');

                fprintf('\n');
                fprintf('Jacobian DOF/node mapping verified.\n');

                fprintf( ...
                    '  Variables-node DOFs     : %d\n', ...
                    xdJ.ndofs);

                fprintf( ...
                    '  Jacobian size           : %d x %d\n', ...
                    size(J_k,1), ...
                    size(J_k,2));

                fprintf( ...
                    '  DOF order               : p,u,v per node\n');

                fprintf( ...
                    '  Nodes                    : %d\n', ...
                    N);

                fprintf( ...
                    '  Reordered nodes          : %d\n', ...
                    num_reordered_nodes);

                fprintf( ...
                    '  Max p/u/v coord error    : %.3e\n', ...
                    max_puv_coord_error);

                fprintf( ...
                    '  Max COMSOL/P match error : %.3e\n', ...
                    max_match_error);

            end


            %% =================================================
            % REORDER JACOBIAN INTO GNN NODE ORDER
            % =================================================

            J_k = J_k( ...
                jacobian_perm, ...
                jacobian_perm);


            %% =================================================
            % SANITY CHECKS
            % =================================================

            if ~issparse(J_k)
                J_k = sparse(J_k);
            end

            if nnz(J_k) == 0

                error( ...
                    'Jacobian at transition %d is empty.', ...
                    k);

            end

            if size(J_k,1) ~= size(J_k,2)

                error( ...
                    ['Jacobian at transition %d is not square: ' ...
                     '%d x %d.'], ...
                    k, ...
                    size(J_k,1), ...
                    size(J_k,2));

            end

            J_nonzero = nonzeros(J_k);

            if any(~isfinite(J_nonzero))

                error( ...
                    ['Jacobian at transition %d contains ' ...
                     'NaN/Inf.'], ...
                    k);

            end


            %% =================================================
            % VERIFY CONSTANT MATRIX DIMENSIONS
            % =================================================

            if isempty(jacobian_shape)

                jacobian_shape = int64([ ...
                    size(J_k,1), ...
                    size(J_k,2)]);


                if jacobian_shape(2) ~= 3*N

                    error( ...
                        ['Jacobian has %d columns, but 3*N = %d. ' ...
                         'The expected p/u/v DOF mapping is ' ...
                         'invalid.'], ...
                        jacobian_shape(2), ...
                        3*N);

                end


                %% --------------------------------------------
                % GNN node -> Jacobian DOF mapping
                % ---------------------------------------------

                node_index_zero = ...
                    int64((0:N-1).');

                jacobian_dof_p = ...
                    3 * node_index_zero;

                jacobian_dof_u = ...
                    3 * node_index_zero + 1;

                jacobian_dof_v = ...
                    3 * node_index_zero + 2;

                jacobian_dof_order = ...
                    'node_interleaved_p_u_v';

                jacobian_node_order = ...
                    'gnn_mesh_P';

                jacobian_permuted_from_comsol = true;


                fprintf('\nJacobian DOF mapping:\n');

                fprintf( ...
                    '  order = %s\n', ...
                    jacobian_dof_order);

                fprintf( ...
                    '  p: %d -> %d\n', ...
                    min(jacobian_dof_p), ...
                    max(jacobian_dof_p));

                fprintf( ...
                    '  u: %d -> %d\n', ...
                    min(jacobian_dof_u), ...
                    max(jacobian_dof_u));

                fprintf( ...
                    '  v: %d -> %d\n', ...
                    min(jacobian_dof_v), ...
                    max(jacobian_dof_v));

                fprintf( ...
                    '  Jacobian columns mapped: %d\n', ...
                    jacobian_shape(2));

            else

                if size(J_k,1) ~= jacobian_shape(1) || ...
                   size(J_k,2) ~= jacobian_shape(2)

                    error( ...
                        ['Jacobian shape changed at transition %d: ' ...
                         '%d x %d instead of %d x %d.'], ...
                        k, ...
                        size(J_k,1), ...
                        size(J_k,2), ...
                        jacobian_shape(1), ...
                        jacobian_shape(2));

                end

            end


            %% =================================================
            % STORE CURRENT JACOBIAN
            % =================================================

            jacobian_sparse{k} = J_k;

        end


    catch ME

        %% ====================================================
        % CLEANUP IF ASSEMBLY FAILS
        % =====================================================

        try
            model.component('comp1').variable.remove(var_tag);
        catch
        end

        try
            model.sol.remove(jacobian_solver_tag);
        catch
        end

        rethrow(ME);

    end


    %% ========================================================
    % VERIFY SOURCE TD SOLUTION WAS NOT MODIFIED
    % ========================================================

    source_info_after = ...
        mphsolinfo(model, 'soltag', solution_tag);

    num_source_solutions_after = ...
        length(source_info_after.solvals);

    if num_source_solutions_after ~= ...
            num_source_solutions_before

        error( ...
            ['Jacobian extraction modified the source TD ' ...
             'solution: %d snapshots before, %d after.'], ...
            num_source_solutions_before, ...
            num_source_solutions_after);

    end

    if any(abs( ...
            source_info_after.solvals(:) ...
            - source_info_before.solvals(:)) > 1e-12)

        error( ...
            ['Jacobian extraction changed the time values ' ...
             'stored in the source TD solution.']);

    end

    fprintf( ...
        '\nSource TD solution preserved: %d snapshots.\n', ...
        num_source_solutions_after);


    %% ========================================================
    % REMOVE TEMPORARY COMSOL OBJECTS
    % ========================================================

    model.component('comp1').variable.remove(var_tag);

    model.sol.remove(jacobian_solver_tag);


    %% ========================================================
    % DIAGNOSTICS
    % ========================================================

    fprintf('\nJacobian extraction complete.\n');

    fprintf( ...
        'Number of Jacobians : %d\n', ...
        num_transitions);

    fprintf( ...
        'Jacobian shape       : %d x %d\n', ...
        jacobian_shape(1), ...
        jacobian_shape(2));


    %% ========================================================
    % CHECK SHARED SPARSITY PATTERN
    % ========================================================

    [row_ref, col_ref, ~] = ...
        find(jacobian_sparse{1});

    shared_pattern = true;

    for k = 2:num_transitions

        [row_k, col_k, ~] = ...
            find(jacobian_sparse{k});

        if length(row_k) ~= length(row_ref) || ...
           ~isequal(row_k, row_ref) || ...
           ~isequal(col_k, col_ref)

            shared_pattern = false;

            fprintf( ...
                ['Jacobian sparsity pattern differs at ' ...
                 'transition %d.\n'], ...
                k);

            break

        end

    end


    %% ========================================================
    % CONVERT TO STORAGE FORMAT
    % ========================================================

    if shared_pattern

        jacobian_shared_pattern = true;

        % Python zero-based indexing
        jacobian_row = ...
            int64(row_ref - 1);

        jacobian_col = ...
            int64(col_ref - 1);

        num_jacobian_nonzeros = ...
            length(row_ref);

        jacobian_values = zeros( ...
            num_transitions, ...
            num_jacobian_nonzeros);

        for k = 1:num_transitions

            [~, ~, values_k] = ...
                find(jacobian_sparse{k});

            jacobian_values(k, :) = ...
                values_k(:).';

        end

    else

        fprintf([ ...
            'Jacobian sparsity pattern is NOT constant.\n' ...
            'Using per-transition COO storage.\n']);

        jacobian_shared_pattern = false;

        jacobian_row = ...
            cell(num_transitions, 1);

        jacobian_col = ...
            cell(num_transitions, 1);

        jacobian_values = ...
            cell(num_transitions, 1);

        for k = 1:num_transitions

            [row_k, col_k, values_k] = ...
                find(jacobian_sparse{k});

            jacobian_row{k} = ...
                int64(row_k - 1);

            jacobian_col{k} = ...
                int64(col_k - 1);

            jacobian_values{k} = ...
                values_k;

        end

    end


    %% ========================================================
    % TARGET ALIGNMENT
    % ========================================================

    jacobian_target_solnum = ...
        int64((2:Nt).');

    jacobian_target_time = ...
        t(2:Nt).';


    %% ========================================================
    % OUTPUT STRUCTURE
    % ========================================================

    jacobian.shared_pattern = ...
        jacobian_shared_pattern;

    jacobian.row = ...
        jacobian_row;

    jacobian.col = ...
        jacobian_col;

    jacobian.values = ...
        jacobian_values;

    jacobian.shape = ...
        jacobian_shape;

    jacobian.target_solnum = ...
        jacobian_target_solnum;

    jacobian.target_time = ...
        jacobian_target_time;

    jacobian.dof_p = ...
        jacobian_dof_p;

    jacobian.dof_u = ...
        jacobian_dof_u;

    jacobian.dof_v = ...
        jacobian_dof_v;

    jacobian.dof_order = ...
        jacobian_dof_order;

    jacobian.node_order = ...
        jacobian_node_order;

    jacobian.permuted_from_comsol = ...
        jacobian_permuted_from_comsol;

end

function validate_dataset( ...
    X, physics_features, geometry_features, ...
    bc, graph, jacobian, t, cfg)

    N  = size(X, 2);
    Nt = size(X, 1);


    %% --------------------------------------------------------
    % Graph checks
    % ---------------------------------------------------------

    if size(graph.edge_index, 1) ~= 2 * graph.Nedges
        error('Unexpected number of directed edges.');
    end

    if length(graph.edge_weight) ~= ...
            size(graph.edge_index, 1)

        error('edge_index and edge_weight are inconsistent.');
    end

    if any(graph.edge_index(:) < 0) || ...
       any(graph.edge_index(:) > N - 1)

        error('Invalid Python node indices in edge_index.');
    end


    %% --------------------------------------------------------
    % Boundary-condition dimensions
    % ---------------------------------------------------------

    if length(bc.velocity_dirichlet_mask) ~= N
        error( ...
            'velocity_dirichlet_mask does not match mesh nodes.');
    end

    if size(bc.velocity_bc, 1) ~= Nt || ...
       size(bc.velocity_bc, 2) ~= N || ...
       size(bc.velocity_bc, 3) ~= 2

        error('velocity_bc has unexpected dimensions.');
    end

    if length(bc.pressure_dirichlet_mask) ~= N
        error( ...
            'pressure_dirichlet_mask does not match mesh nodes.');
    end

    if size(bc.pressure_bc, 1) ~= Nt || ...
       size(bc.pressure_bc, 2) ~= N

        error('pressure_bc has unexpected dimensions.');
    end


    %% --------------------------------------------------------
    % Finite-value checks
    % ---------------------------------------------------------

    if any(~isfinite(X(:)))
        error('X contains NaN or Inf.');
    end

    if any(~isfinite(physics_features(:)))
        error('physics_features contains NaN or Inf.');
    end

    if any(~isfinite(geometry_features(:)))
        error('geometry_features contains NaN or Inf.');
    end

    if any(~isfinite(bc.velocity_bc(:)))
        error('velocity_bc contains NaN or Inf.');
    end

    if any(~isfinite(bc.pressure_bc(:)))
        error('pressure_bc contains NaN or Inf.');
    end


    %% --------------------------------------------------------
    % Jacobian alignment
    % ---------------------------------------------------------

    if cfg.export_jacobian

        if length(jacobian.target_time) ~= Nt - 1

            error( ...
                ['Number of Jacobians does not match ' ...
                 'number of transitions.']);

        end

        if any(abs( ...
                jacobian.target_time(:) ...
                - t(2:end).') > 1e-12)

            error( ...
                ['Jacobian target times are misaligned ' ...
                 'with X targets.']);

        end

    end


    %% --------------------------------------------------------
    % Prescribed velocity BC vs COMSOL solution
    % ---------------------------------------------------------

    bc_difference = abs( ...
        bc.velocity_bc(:, bc.velocity_dirichlet_mask, :) ...
        - X(:, bc.velocity_dirichlet_mask, 1:2));

    max_bc_difference = max(bc_difference(:));

    velocity_scale = ...
        max(abs(X(:,:,1:2)), [], 'all');

    fprintf( ...
        '\nPrescribed BC vs solution at the %d Dirichlet nodes:\n', ...
        sum(bc.velocity_dirichlet_mask));

    fprintf( ...
        ['  largest disagreement: %.6e ' ...
         '(%.4f%% of max |velocity|)\n'], ...
        max_bc_difference, ...
        100 * max_bc_difference / velocity_scale);


    if max_bc_difference > ...
            1e-6 * max(velocity_scale, 1)

        node_difference = squeeze( ...
            max(max(bc_difference, [], 1), [], 3));

        disagreeing = find( ...
            node_difference > ...
            1e-6 * max(velocity_scale, 1));

        dirichlet_indices = ...
            find(bc.velocity_dirichlet_mask);

        disagreeing_nodes = ...
            dirichlet_indices(disagreeing);

        warning( ...
            ['The prescribed velocity BC and the COMSOL solution ' ...
             'disagree at %d of %d Dirichlet nodes ' ...
             '(largest %.3e). Of these, %d are shared ' ...
             'wall/inlet nodes.'], ...
            numel(disagreeing_nodes), ...
            sum(bc.velocity_dirichlet_mask), ...
            max_bc_difference, ...
            sum(bc.shared_wall_inlet_nodes(disagreeing_nodes)));

    end


    %% --------------------------------------------------------
    % Prescribed pressure BC vs COMSOL solution
    % ---------------------------------------------------------

    pressure_difference = abs( ...
        bc.pressure_bc(:, bc.pressure_dirichlet_mask) ...
        - X(:, bc.pressure_dirichlet_mask, 3));

    max_pressure_bc_difference = ...
        max(pressure_difference(:));

    pressure_scale = ...
        max(abs(X(:,:,3)), [], 'all');

    fprintf( ...
        ['\nPrescribed pressure BC vs solution at the ' ...
         '%d Dirichlet nodes:\n'], ...
        sum(bc.pressure_dirichlet_mask));

    fprintf( ...
        '  largest disagreement: %.6e\n', ...
        max_pressure_bc_difference);

    if max_pressure_bc_difference > ...
            1e-6 * max(pressure_scale, 1)

        warning( ...
            ['The prescribed pressure BC and the COMSOL solution ' ...
             'disagree at pressure Dirichlet nodes. ' ...
             'Largest disagreement: %.3e.'], ...
            max_pressure_bc_difference);

    end

end

function save_dataset( ...
    cfg, ...
    X, ...
    physics_features, physics_feature_names, ...
    geometry_features, geometry_feature_names, ...
    bc, graph, ...
    T, P, t, ...
    G_wlsq, neighbors_python, ...
    U_inf, fluid, jacobian)

    %% --------------------------------------------------------
    % Recover original variable names expected by Python
    % ---------------------------------------------------------

    wall_mask = bc.wall_mask;
    inlet_mask = bc.inlet_mask;
    outlet_mask = bc.outlet_mask;

    velocity_dirichlet_mask = ...
        bc.velocity_dirichlet_mask;

    velocity_bc = ...
        bc.velocity_bc;

    pressure_dirichlet_mask = ...
        bc.pressure_dirichlet_mask;

    pressure_bc = ...
        bc.pressure_bc;


    edge_index = graph.edge_index;
    edge_weight = graph.edge_weight;
    cell_index = graph.cell_index;
    h = graph.h;


    %% --------------------------------------------------------
    % Save base dataset
    % ---------------------------------------------------------

    save(cfg.output_file, ...
        'X', ...
        'physics_features', ...
        'physics_feature_names', ...
        'geometry_features', ...
        'geometry_feature_names', ...
        'wall_mask', ...
        'inlet_mask', ...
        'outlet_mask', ...
        'velocity_dirichlet_mask', ...
        'velocity_bc', ...
        'pressure_dirichlet_mask', ...
        'pressure_bc', ...
        'edge_index', ...
        'edge_weight', ...
        'T', ...
        'cell_index', ...
        'neighbors_python', ...
        'G_wlsq', ...
        'P', ...
        't', ...
        'h', ...
        'U_inf', ...
        '-v7.3');


    %% --------------------------------------------------------
    % Fluid properties
    % ---------------------------------------------------------

    if fluid.export

        rho = fluid.rho;
        mu = fluid.mu;

        save( ...
            cfg.output_file, ...
            'rho', ...
            'mu', ...
            '-append');

    end


    %% --------------------------------------------------------
    % FEM Jacobians
    % ---------------------------------------------------------

    if cfg.export_jacobian

        jacobian_shared_pattern = ...
            jacobian.shared_pattern;

        jacobian_row = ...
            jacobian.row;

        jacobian_col = ...
            jacobian.col;

        jacobian_values = ...
            jacobian.values;

        jacobian_shape = ...
            jacobian.shape;

        jacobian_target_solnum = ...
            jacobian.target_solnum;

        jacobian_target_time = ...
            jacobian.target_time;

        jacobian_dof_p = ...
            jacobian.dof_p;

        jacobian_dof_u = ...
            jacobian.dof_u;

        jacobian_dof_v = ...
            jacobian.dof_v;

        jacobian_dof_order = ...
            jacobian.dof_order;

        jacobian_node_order = ...
            jacobian.node_order;

        jacobian_permuted_from_comsol = ...
            jacobian.permuted_from_comsol;


        % Keep original metadata names in the MAT file
        jacobian_matrix_type = ...
            cfg.jacobian_matrix_type;

        jacobian_rowscale = ...
            cfg.jacobian_rowscale;

        jacobian_linearization = ...
            cfg.jacobian_linearization;


        save(cfg.output_file, ...
            'jacobian_shared_pattern', ...
            'jacobian_row', ...
            'jacobian_col', ...
            'jacobian_values', ...
            'jacobian_shape', ...
            'jacobian_target_solnum', ...
            'jacobian_target_time', ...
            'jacobian_dof_p', ...
            'jacobian_dof_u', ...
            'jacobian_dof_v', ...
            'jacobian_dof_order', ...
            'jacobian_node_order', ...
            'jacobian_permuted_from_comsol', ...
            'jacobian_matrix_type', ...
            'jacobian_rowscale', ...
            'jacobian_linearization', ...
            '-append');

        fprintf( ...
            'FEM Jacobians appended successfully.\n');

    end


    fprintf( ...
        '\nDataset saved successfully:\n%s\n', ...
        cfg.output_file);

end