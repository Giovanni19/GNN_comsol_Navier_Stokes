
function [n_iter, info] = get_nonlinear_iterations(log_file)
% ============================================================
% GET NONLINEAR ITERATIONS
%
% Reads the LAST nonlinear-solver iteration table contained
% in a COMSOL progress log and returns the number of nonlinear
% iterations performed in that solve.
%
% The function:
%   1. Finds all "Nonlinear solver" sections.
%   2. Selects the last one.
%   3. Finds the corresponding iteration-table header.
%   4. Reads all nonlinear iteration rows, ignoring COMSOL
%      progress and memory messages inserted inside the table.
%   5. Verifies that the iteration numbering is consistent.
%
% INPUT
%   log_file : path to the COMSOL solver log
%
% OUTPUT
%   n_iter   : number of nonlinear iterations
%
% ============================================================


%% Check input file

if ~isfile(log_file)
    error('COMSOL solver log not found: %s', log_file);
end


%% Read log

log_text = fileread(log_file);

if isempty(log_text)
    error('COMSOL solver log is empty: %s', log_file);
end


%% Split into lines

lines = splitlines(log_text);


%% Find all "Nonlinear solver" sections

nonlinear_idx = find(contains(lines, 'Nonlinear solver'));

if isempty(nonlinear_idx)
    error('No "Nonlinear solver" section found in COMSOL log.');
end


%% Select the last nonlinear-solver section

start_line = nonlinear_idx(end);


%% Define the end of the section

% We selected the last "Nonlinear solver" section in the log,
% therefore we can safely scan until the end of the file.
% The iteration parser will stop when it reaches the end of the
% Stationary Solver section.

end_line = length(lines);

%% Find iteration-table header
%
% We require the first non-space characters to be "Iter".
% This is stricter than simply searching for the word anywhere
% in the line.

header_line = [];

for i = start_line:end_line

    line = strtrim(lines{i});

    if ~isempty(regexp(line, '^Iter(\s|$)', 'once'))
        header_line = i;
        break;
    end

end

if isempty(header_line)
    error(['Nonlinear iteration table header not found after ' ...
           'the last "Nonlinear solver" section.']);
end


%% Read nonlinear iteration rows
%
% COMSOL may insert progress-information lines inside the nonlinear
% iteration table, for example:
%
%   1   ...
%   2   ...
%   -- Current Progress: 27 % -
%   Memory: ...
%   3   ...
%
% Therefore iteration rows are NOT necessarily consecutive.
%
% We continue scanning after the table header and collect every
% line that starts with an integer followed by whitespace.
%
% The scan stops when the Stationary Solver section ends.

iteration_numbers = [];
iteration_lines = strings(0,1);

for i = header_line + 1:end_line

    line = strtrim(lines{i});


    % ---------------------------------------------------------
    % Detect the end of the Stationary Solver section
    % ---------------------------------------------------------
    %
    % COMSOL closes a solver section with a line such as:
    %
    % ----- Stationary Solver 1 in Study 3/Solution 3 (sol3) ---->
    %
    % Once this line is reached, there cannot be any more
    % nonlinear iterations belonging to this solve.

    if contains(line, '----- Stationary Solver') && ...
            contains(line, '----->')

        break;

    end


    % ---------------------------------------------------------
    % Detect nonlinear iteration rows
    % ---------------------------------------------------------
    %
    % A nonlinear iteration row starts with an integer:
    %
    %   1    71      1.5e+03 ...
    %   2    6.2     1.4e+03 ...
    %
    % Progress lines, memory information, etc. are ignored.

    token = regexp(line, '^(\d+)\s+', 'tokens', 'once');


    if ~isempty(token)

        iteration_number = str2double(token{1});

        if ~isfinite(iteration_number)

            error( ...
                'Invalid nonlinear iteration number in COMSOL log.');

        end


        iteration_numbers(end+1,1) = iteration_number;
        iteration_lines(end+1,1) = string(line);

    end

end


%% Validate extracted iterations

if isempty(iteration_numbers)
    error('No nonlinear iteration rows found in COMSOL log.');
end


% Iteration numbers should increase monotonically.
if any(diff(iteration_numbers) <= 0)
    error(['Nonlinear iteration numbers are not strictly increasing. ' ...
           'The COMSOL log may have been parsed incorrectly.']);
end


% Iteration numbering should normally be consecutive.
if length(iteration_numbers) > 1

    if any(diff(iteration_numbers) ~= 1)
        error(['Nonlinear iteration numbering is not consecutive. ' ...
               'The COMSOL log may contain an unexpected format.']);
    end

end


%% Return number of nonlinear iterations
%
% We use the number of actual iteration rows rather than simply
% max(iteration_numbers). This makes explicit what we are
% measuring: the number of nonlinear iteration records present
% in the COMSOL table.

n_iter = length(iteration_numbers);

%% Parse nonlinear solver table

% Read the column names directly from the COMSOL header.
% This allows the parser to work whether COMSOL prints
% SolEst only, ResEst only, or both.

header = strtrim(lines{header_line});
column_names = regexp(header, '\S+', 'match');

num_columns = length(column_names);
num_rows = length(iteration_lines);

table_data = nan(num_rows, num_columns);

for j = 1:num_rows

    values = sscanf(char(iteration_lines(j)), '%f');

    if length(values) ~= num_columns
        error( ...
            ['Unexpected COMSOL nonlinear solver row format.\n' ...
             'Header contains %d columns, row contains %d values.\n' ...
             'Header: %s\n' ...
             'Row: %s'], ...
             num_columns, ...
             length(values), ...
             header, ...
             iteration_lines(j));
    end

    table_data(j,:) = values(:).';

end


%% Locate columns from COMSOL header

idx_iter     = find(strcmp(column_names, 'Iter'), 1);
idx_sol_est  = find(strcmp(column_names, 'SolEst'), 1);
idx_res_est  = find(strcmp(column_names, 'ResEst'), 1);
idx_damping  = find(strcmp(column_names, 'Damping'), 1);
idx_stepsize = find(strcmp(column_names, 'Stepsize'), 1);

idx_res = find(strcmp(column_names, '#Res'), 1);
idx_jac = find(strcmp(column_names, '#Jac'), 1);
idx_sol = find(strcmp(column_names, '#Sol'), 1);

idx_lin_err = find(strcmp(column_names, 'LinErr'), 1);
idx_lin_res = find(strcmp(column_names, 'LinRes'), 1);


%% Verify required columns

if isempty(idx_iter) || isempty(idx_res) || ...
        isempty(idx_jac) || isempty(idx_sol)

    error( ...
        ['Required nonlinear solver columns not found.\n' ...
         'Header: %s'], ...
         header);

end


%% Verify parsed iteration column

if ~isequal(table_data(:,idx_iter), iteration_numbers)

    error( ...
        ['Parsed Iter column does not match the iteration numbers ' ...
         'identified from the COMSOL log.']);

end


%% Build diagnostic information

info = struct();

info.log_file = log_file;
info.section_start_line = start_line;
info.header_line = header_line;
info.header = header;

info.column_names = column_names;
info.table_data = table_data;

info.iteration_numbers = iteration_numbers;
info.iteration_lines = iteration_lines;

info.first_iteration = iteration_numbers(1);
info.last_iteration = iteration_numbers(end);
info.number_of_rows = length(iteration_numbers);


%% Store parsed nonlinear solver data

info.iteration = table_data(:,idx_iter);

if ~isempty(idx_sol_est)
    info.sol_est = table_data(:,idx_sol_est);
else
    info.sol_est = [];
end

if ~isempty(idx_res_est)
    info.res_est = table_data(:,idx_res_est);
else
    info.res_est = [];
end

if ~isempty(idx_damping)
    info.damping = table_data(:,idx_damping);
else
    info.damping = [];
end

if ~isempty(idx_stepsize)
    info.stepsize = table_data(:,idx_stepsize);
else
    info.stepsize = [];
end


%% Solver counters

info.num_res = table_data(:,idx_res);
info.num_jac = table_data(:,idx_jac);
info.num_sol = table_data(:,idx_sol);


%% Optional linear solver information

if ~isempty(idx_lin_err)
    info.lin_err = table_data(:,idx_lin_err);
else
    info.lin_err = [];
end

if ~isempty(idx_lin_res)
    info.lin_res = table_data(:,idx_lin_res);
else
    info.lin_res = [];
end


%% Final cumulative solver statistics

info.total_residual_evaluations = info.num_res(end);
info.total_jacobian_evaluations = info.num_jac(end);
info.total_linear_solves        = info.num_sol(end);
%% Additional consistency check

expected_last_iteration = ...
    iteration_numbers(1) + n_iter - 1;

if iteration_numbers(end) ~= expected_last_iteration

    error( ...
        ['Internal consistency check failed. ' ...
         'First iteration: %d, last iteration: %d, rows found: %d.'], ...
         iteration_numbers(1), ...
         iteration_numbers(end), ...
         n_iter);

end

end

