import os
import json
import glob
import numpy as np
from pathlib import Path
import re

# Define parameter extraction patterns for each method
METHOD_PARAM_PATTERNS = {
    'pcmci':['taumax', 'pcalpha', 'alpha'],
    'varlingam':['taumax', 'varlingamalpha'],
    'dynotears':['taumax', 'wthre', 'lambda_a', 'lambda_w'],
    'ntsnotears':['taumax', 'wthre', 'lambda_1', 'lambda_2'],
    'cmlp':['lam', 'lr'],
    'clstm':['lam', 'lr'],
    'cutsplus':['input_step', 'batchsize', 'weight_decay'],
    'cuts':['input_step', 'batchsize', 'weight_decay'],
    'lgc':['taumax', 'threshold', 'alphas'],
    'pwgc':['alpha', 'taumax'],
    'var':['taumax', 'threshold'],
    'tsci':['theta', 'fnn_tol'],
}


def parse_params_from_filename(filename, method):
    """
    Parse parameters from filename using method-specific patterns.
    """
    # Remove method name prefix and .json suffix
    param_str = filename.replace(f"{method}_", "").replace(".json", "")

    params = {}

    # Get parameter names for this method
    param_names = METHOD_PARAM_PATTERNS.get(method, [])

    # For each expected parameter, try to extract its value
    for param_name in param_names:
        # Build regex pattern to extract value
        # Use word boundary or underscore to ensure we match the full parameter name
        # (?:^|_) means "start of string or underscore"
        pattern = rf'(?:^|_){re.escape(param_name)}_?([0-9.eE+-]+(?:_[0-9.eE+-]+)*)'
        match = re.search(pattern, param_str)

        if match:
            value_str = match.group(1)

            # Handle multiple values separated by underscore (like lgc alphas)
            if '_' in value_str and param_name == 'alphas':
                values = []
                for v in value_str.split('_'):
                    try:
                        if '.' in v or 'e' in v.lower():
                            values.append(float(v))
                        else:
                            values.append(int(v))
                    except ValueError:
                        continue
                if values:
                    params[param_name] = tuple(values)

            # Handle ntsnotears lambda_1 which might have underscore separator
            elif '_' in value_str and param_name in ['lambda_1', 'lambda_2']:
                # For Lorenz: lambda_1 might be "0.001_0.1" (two values)
                # We need to convert this to a tuple for matching
                if param_name == 'lambda_1' and value_str.count('_') >= 1:
                    # This might be a list format like "0.001_0.1"
                    # Store as tuple for exact matching
                    values = []
                    for v in value_str.split('_'):
                        try:
                            if '.' in v or 'e' in v.lower():
                                values.append(float(v))
                            else:
                                values.append(int(v))
                        except ValueError:
                            continue
                    if values:
                        # If only one value, store as scalar; if multiple, store as tuple
                        params[param_name] = tuple(values) if len(values) > 1 else values[0]
                else:
                    # Regular single value (lambda_2)
                    actual_value = value_str.split('_')[-1]
                    try:
                        if '.' in actual_value or 'e' in actual_value.lower():
                            params[param_name] = float(actual_value)
                        else:
                            params[param_name] = int(actual_value)
                    except ValueError:
                        continue

            # Regular single value
            else:
                try:
                    if '.' in value_str or 'e' in value_str.lower():
                        params[param_name] = float(value_str)
                    else:
                        params[param_name] = int(value_str)
                except ValueError:
                    continue

    return params

def check_constraints(params, constraints):
    """
    Check if parameters satisfy constraints.

    Parameters:
    -----------
    params : dict
        Parameter name-value pairs from filename
    constraints : dict
        Constraints for each parameter
        Format:
        - Single value: {'taumax': 5}
        - Tuple (range): {'alpha': (0.01, 0.1)}
        - List (allowed values): {'taumax': [3, 5, 7]}

    Returns:
    --------
    bool : True if all constraints are satisfied
    """
    for param_name, constraint in constraints.items():
        if param_name not in params:
            return False

        param_value = params[param_name]

        # Constraint is a dict with explicit type
        if isinstance(constraint, dict):
            constraint_type = constraint.get('type', 'exact')

            if constraint_type == 'exact':
                if param_value != constraint['value']:
                    return False

            elif constraint_type == 'range':
                min_val = constraint.get('min', float('-inf'))
                max_val = constraint.get('max', float('inf'))
                if not (min_val <= param_value <= max_val):
                    return False

            elif constraint_type == 'values':
                allowed = constraint['allowed']
                if param_value not in allowed:
                    return False

        # Constraint is a tuple -> range (ONLY if exactly 2 elements AND both are numbers)
        elif isinstance(constraint, tuple):
            if len(constraint) == 2:
                # Check if both elements are numbers (int or float)
                if all(isinstance(x, (int, float)) for x in constraint):
                    # This is a range constraint
                    min_val, max_val = constraint
                    if not (min_val <= param_value <= max_val):
                        return False
                else:
                    # Not a range, treat as exact match
                    if param_value != constraint:
                        return False
            else:
                # More than 2 elements or 1 element, treat as exact match
                if param_value != constraint:
                    return False

        # Constraint is a list -> allowed values
        elif isinstance(constraint, list):
            if param_value not in constraint:
                return False

        # Constraint is a single value -> exact match
        else:
            if param_value != constraint:
                return False

    return True

def load_best_result(result_dir, method, param_constraints=None):
    """
    Load the best result (highest AUPRC) for a given method with parameter constraints.

    Parameters:
    -----------
    result_dir : str
        Directory containing result JSON files
    method : str
        Method name (e.g., 'var', 'pcmci')
    param_constraints : dict, optional
        Parameter constraints for filtering files

    Returns:
    --------
    dict : Best result or None
    """
    if not os.path.exists(result_dir):
        return None

    pattern = os.path.join(result_dir, f"{method}_*.json")
    files = glob.glob(pattern)

    if not files:
        return None

    # Filter files based on parameter constraints
    valid_files = []
    if param_constraints is not None:
        for file in files:
            filename = os.path.basename(file)
            params = parse_params_from_filename(filename, method)

            if check_constraints(params, param_constraints):
                valid_files.append(file)
    else:
        valid_files = files

    if not valid_files:
        return None

    best_auprc = -1
    best_result = None

    for file in valid_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                auroc = data['performance']['auroc_no_diag_mean']
                auprc = data['performance']['auprc_no_diag_mean']

                # Compare AUPRC first (primary metric), then AUROC if tied
                if auprc > best_auprc:
                    # Better AUPRC, update best result
                    best_auprc = auprc
                    best_result = {
                        'auroc_mean': data['performance']['auroc_no_diag_mean'],
                        'auroc_std': data['performance']['auroc_no_diag_std'],
                        'auprc_mean': data['performance']['auprc_no_diag_mean'],
                        'auprc_std': data['performance']['auprc_no_diag_std'],
                        'file': os.path.basename(file)
                    }
                elif auprc == best_auprc and best_result is not None:
                    # Same AUPRC, compare AUROC (tie-breaker)
                    if auroc > best_result['auroc_mean']:
                        # Better AUROC, update best result
                        best_auprc = auprc
                        best_result = {
                            'auroc_mean': data['performance']['auroc_no_diag_mean'],
                            'auroc_std': data['performance']['auroc_no_diag_std'],
                            'auprc_mean': data['performance']['auprc_no_diag_mean'],
                            'auprc_std': data['performance']['auprc_no_diag_std'],
                            'file': os.path.basename(file)
                        }
        except (KeyError, json.JSONDecodeError):
            continue

    return best_result


    # # todo: Compare AUROC first, then AUPRC if AUROC is equal
    # best_auroc = -1
    # best_result = None
    #
    # for file in valid_files:
    #     try:
    #         with open(file, 'r') as f:
    #             data = json.load(f)
    #             auroc = data['performance']['auroc_no_diag_mean']
    #             auprc = data['performance']['auprc_no_diag_mean']
    #
    #             # Compare AUROC first, then AUPRC if AUROC is equal
    #             if auroc > best_auroc:
    #                 # Better AUROC, update best result
    #                 best_auroc = auroc
    #                 best_result = {
    #                     'auroc_mean':data['performance']['auroc_no_diag_mean'],
    #                     'auroc_std':data['performance']['auroc_no_diag_std'],
    #                     'auprc_mean':data['performance']['auprc_no_diag_mean'],
    #                     'auprc_std':data['performance']['auprc_no_diag_std'],
    #                     'file':os.path.basename(file)
    #                 }
    #             elif auroc == best_auroc and best_result is not None:
    #                 # Same AUROC, compare AUPRC
    #                 if auprc > best_result['auprc_mean']:
    #                     # Better AUPRC, update best result
    #                     best_auroc = auroc
    #                     best_result = {
    #                         'auroc_mean':data['performance']['auroc_no_diag_mean'],
    #                         'auroc_std':data['performance']['auroc_no_diag_std'],
    #                         'auprc_mean':data['performance']['auprc_no_diag_mean'],
    #                         'auprc_std':data['performance']['auprc_no_diag_std'],
    #                         'file':os.path.basename(file)
    #                     }
    #     except (KeyError, json.JSONDecodeError):
    #         continue
    #
    # return best_result

def format_value(mean, std, is_best=False):
    """
    Format mean±std as percentage with 1 decimal place.

    Parameters:
    -----------
    mean : float
        Mean value (0-1 range)
    std : float
        Standard deviation (0-1 range)
    is_best : bool
        Whether to make the value bold

    Returns:
    --------
    str : Formatted LaTeX string
    """
    mean_pct = mean * 100
    std_pct = std * 100

    # Truncate to 1 decimal place (not round)
    mean_str = f"{int(mean_pct * 10) / 10:.1f}"
    std_str = f"{int(std_pct * 10) / 10:.1f}"

    formatted = f"{mean_str}\\footnotesize{{$\\pm${std_str}}}"

    if is_best:
        formatted = f"\\textbf{{{formatted}}}"

    return formatted


def get_scenario_display_name(scenario):
    """
    Map scenario internal name to display name.

    Parameters:
    -----------
    scenario : str
        Internal scenario name (e.g., 'confounder_rho0.5')

    Returns:
    --------
    str : Display name for LaTeX table
    """
    name_mapping = {
        'vanilla':'Vanilla',
        'confounder_rho0.5':'Latent confounders',
        'measurement_error_gamma10.0':'Measurement error',
        'measurement_error_gamma1.2':'Measurement error',
        'iid':'IID',
        'trendseason':'Trend and seasonality',
        'standardized_zscore':'Standardized',
        'standardized_minmax':'Min--max normalization',
        'missing_prob0.2':'Missing',
        'missing_prob0.4':'Missing',
        'mixed_data_ratio0.5':'Mixed data',
        'mixed_noise_gaussian0.5':'Mixed noise',
        'nonstationary_noisestd1.0_mean1.0':'Nonstationary',  # VAR
        'nonstationary_noisestd2.0_mean2.5':'Nonstationary',  # Lorenz F=10
        'nonstationary_noisestd2.0_mean3.5':'Nonstationary',  # Lorenz F=40
        'mechanism_violation':'Mechanism violation'
    }
    return name_mapping.get(scenario, scenario)


def get_method_display_name(method):
    """Map method internal name to display name."""
    mapping = {
        'ntsnotears':'NTS-NOTEARS',
        'cmlp':'cMLP',
        'clstm':'cLSTM',
        'cutsplus':'CUTS+',
        'varlingam':'VARLiNGAM'
    }
    return mapping.get(method, method.upper())


def generate_single_table(data_model, p, T, F, methods, scenarios, results,
                          best_auroc, best_auprc, part_num):
    """
    Generate a single LaTeX table (Part I or Part II).

    Parameters:
    -----------
    data_model : str
        'VAR' or 'Lorenz'
    p : int
        Number of variables
    T : int
        Time series length
    F : float or None
        Forcing parameter for Lorenz
    methods : list of str
        List of method names
    scenarios : list of str
        List of scenario names for this table
    results : dict
        Nested dictionary with all results
    best_auroc : dict
        Best AUROC values per scenario
    best_auprc : dict
        Best AUPRC values per scenario
    part_num : str
        Part number ('I' or 'II')

    Returns:
    --------
    str : LaTeX code for this table
    """
    latex_lines = []

    # Table header
    latex_lines.append("\\begin{table*}[!htbp]")
    latex_lines.append("\t\\centering")

    # Generate caption based on data model
    if data_model == 'VAR':
        caption = f"Linear Setting, {p}-node case with $T = {T}$ (Part {part_num})."
    elif data_model == 'Lorenz':
        if F is not None:
            caption = f"Nonlinear Setting, {p}-node case with $T = {T}$ and $F = {F}$ (Part {part_num})."
        else:
            caption = f"Nonlinear Setting, {p}-node case with $T = {T}$ (Part {part_num})."
    else:
        caption = f"Results for {data_model}, p={p}, T={T} (Part {part_num})."

    latex_lines.append(f"\t\\caption{{{caption}}}")

    if part_num == 'I':
        latex_lines.append("\t\\setlength{\\tabcolsep}{8.3pt}")
    else:  # Part II
        latex_lines.append("\t\\setlength{\\tabcolsep}{4pt}")

    # Column specification
    n_scenarios = len(scenarios)
    col_spec = "c|" + "cc|" * n_scenarios
    col_spec = col_spec.rstrip("|")

    latex_lines.append(f"\t\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\t\t\\toprule")

    # Multi-column header for scenarios
    header_line = "\t\t& "
    for i, scenario in enumerate(scenarios):
        display_name = get_scenario_display_name(scenario)
        if i < len(scenarios) - 1:
            header_line += f"\\multicolumn{{2}}{{c|}}{{{display_name}}} & "
        else:
            header_line += f"\\multicolumn{{2}}{{c}}{{{display_name}}} & "
    header_line = header_line.rstrip("& ")
    header_line += " \\\\"
    latex_lines.append(header_line)

    # Sub-header for AUROC and AUPRC (with upward arrows)
    subheader_line = f"\t\t{p} nodes & "
    for scenario in scenarios:
        subheader_line += "AUROC$\\uparrow$ & AUPRC$\\uparrow$ & "
    subheader_line = subheader_line.rstrip("& ")
    subheader_line += " \\\\"
    latex_lines.append(subheader_line)
    latex_lines.append("\t\t\\midrule")

    # Data rows
    for method in methods:
        display_method = get_method_display_name(method)
        row_line = f"\t\t{display_method} & "

        for scenario in scenarios:
            result = results[scenario][method]

            if result is None:
                # No result available
                row_line += "- & - & "
            else:
                # Check if this is the best value
                is_best_auroc = (result['auroc_mean'] == best_auroc.get(scenario, -1))
                is_best_auprc = (result['auprc_mean'] == best_auprc.get(scenario, -1))

                auroc_str = format_value(result['auroc_mean'], result['auroc_std'], is_best_auroc)
                auprc_str = format_value(result['auprc_mean'], result['auprc_std'], is_best_auprc)

                row_line += f"{auroc_str} & {auprc_str} & "

        row_line = row_line.rstrip("& ")
        row_line += " \\\\"
        latex_lines.append(row_line)

    # Table footer
    latex_lines.append("\t\t\\bottomrule")
    latex_lines.append("\t\\end{tabular}")
    latex_lines.append("\\end{table*}")

    return "\n".join(latex_lines)


def generate_latex_tables(data_model, p, T, F=None, methods=None, scenarios=None,
                          method_param_constraints=None, base_dir='results/synthetic',
                          output_file=None):
    """
    Generate two LaTeX tables (Part I and Part II) for experimental results.

    Parameters:
    -----------
    data_model : str
        'VAR' or 'Lorenz'
    p : int
        Number of variables
    T : int
        Time series length
    F : float, optional
        Forcing parameter for Lorenz (required if data_model='Lorenz')
    methods : list of str
        List of method names (e.g., ['pwgc', 'var', 'lgc', 'pcmci'])
    scenarios : list of str
        List of scenario names (will be split into two tables)
    method_param_constraints : dict, optional
        Parameter constraints for each method
        Format: {method_name: {param_name: constraint, ...}, ...}
        Examples:
        {
            'pcmci': {
                'taumax': [3, 5],           # List: only taumax=3 or taumax=5
                'alpha': (0.01, 0.05)        # Tuple: 0.01 <= alpha <= 0.05
            },
            'var': {
                'taumax': 5,                 # Single value: exact match
                'threshold': [0.05, 0.1, 0.2]  # List: threshold in [0.05, 0.1, 0.2]
            },
            'cutsplus': {
                'input_step': [5, 10],
                'batchsize': (16, 64),        # Range: 16 <= batchsize <= 64
                'weight_decay': 0.001         # Exact value
            }
        }
    base_dir : str
        Base directory for results
    output_file : str, optional
        Output LaTeX file path. If None, prints to stdout

    Returns:
    --------
    str : Generated LaTeX code
    """

    # Default methods if not specified
    if methods is None:
        methods = ['pwgc', 'var', 'lgc', 'varlingam', 'pcmci', 'dynotears', 'ntsnotears', 'tsci', 'cmlp', 'clstm', 'cuts',
               'cutsplus']

    # Default scenarios if not specified (9 scenarios total for current experiments)
    if scenarios is None:
        scenarios = [
            'vanilla',
            'confounder_rho0.5',
            'measurement_error_gamma1.2',
            'iid',
            'trendseason',
            'standardized_zscore',
            'standardized_minmax',
            'missing_prob0.4',
            'mixed_data_ratio0.5',
            'mixed_noise_gaussian0.5',
            'nonstationary_noisestd0.8',
            'mechanism_violation'
        ]

    # Split scenarios into two parts (4 + 5)
    # Part I: 4 columns, Part II: 5 columns
    mid_point = 4  # Fixed split: first 4 scenarios in Part I
    scenarios_part1 = scenarios[:mid_point]
    scenarios_part2 = scenarios[mid_point:]

    # Collect all results
    results = {}

    for scenario in scenarios:
        results[scenario] = {}

        # Construct dimension string based on data model and scenario
        if data_model == 'Lorenz' and scenario == 'mechanism_violation':
            dim_str = f"p{p}_T{T}"
        else:
            dim_str = f"p{p}_T{T}"
            if data_model == 'Lorenz' and F is not None:
                dim_str += f"_F{F}"

        result_dir = os.path.join(base_dir, data_model, scenario, dim_str)

        for method in methods:
            # Get parameter constraints for this method
            param_constraints = None
            if method_param_constraints is not None and method in method_param_constraints:
                param_constraints = method_param_constraints[method]

            result = load_best_result(result_dir, method, param_constraints)
            results[scenario][method] = result

    # Find best values for each scenario and metric
    best_auroc = {}
    best_auprc = {}

    for scenario in scenarios:
        auroc_values = []
        auprc_values = []

        for method in methods:
            if results[scenario][method] is not None:
                auroc_values.append(results[scenario][method]['auroc_mean'])
                auprc_values.append(results[scenario][method]['auprc_mean'])

        if auroc_values:
            best_auroc[scenario] = max(auroc_values)
            best_auprc[scenario] = max(auprc_values)

    # Generate Part I
    table_part1 = generate_single_table(
        data_model, p, T, F, methods, scenarios_part1,
        results, best_auroc, best_auprc, part_num='I'
    )

    # Generate Part II
    table_part2 = generate_single_table(
        data_model, p, T, F, methods, scenarios_part2,
        results, best_auroc, best_auprc, part_num='II'
    )

    # Combine both tables
    latex_code = table_part1 + "\n\n" + table_part2

    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_code)
        print(f"LaTeX tables saved to: {output_file}")
    else:
        print(latex_code)

    return latex_code


# Example usage
if __name__ == "__main__":

    param_constraints_VAR = {
        'pwgc':{'taumax':[3, 5], 'alpha':[0.01, 0.05, 0.1]},
        'var':{'taumax':[3, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3]},      # 'var':{'taumax':[1, 2, 3, 4, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3]},
        'lgc':{'taumax':[3, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3], 'alphas':(1e-4, 5e-3, 1e-2, 2e-2, 5e-2)},      #'lgc':{'taumax':[1, 2, 3, 4, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3], 'alphas':(1e-4, 5e-3, 1e-2, 2e-2, 5e-2)},
        'pcmci':{'taumax':[3, 5], 'alpha':[0.01, 0.05, 0.1]},
        'varlingam':{'taumax':[3, 5], 'varlingamalpha':[0, 0.01, 0.05, 0.1, 0.3]},
        'dynotears':{'taumax':[3, 5], 'wthre':[0.01, 0.05, 0.1, 0.3],
                     'lambda_a':[0.001, 0.01, 0.1], 'lambda_w':[0.001, 0.005, 0.01]},
        'tsci':{'theta':[0.4, 0.5, 0.6], 'fnn_tol':[0.005, 0.01]},
        'cuts':{'input_step':[1, 3, 5, 10], 'batchsize':[32, 128], 'weight_decay':[0, 0.001, 0.003]},
        'cutsplus':{'input_step':[1, 3, 5, 10], 'batchsize':[32, 128], 'weight_decay':[0, 0.001, 0.003]},

        'ntsnotears':{
            'taumax':[3],                                     # 'taumax':[1, 2, 3, 4, 5],
            'wthre':[0.01, 0.05, 0.1, 0.3, 0.5],
            'lambda_1':[0.001, 0.01, 0.1],
            'lambda_2':[0.001, 0.005, 0.01]
        },

        'cmlp':{
            'lam':[0.0001, 0.005, 0.05],
            'lr':[0.01, 0.1]
        },

        'clstm':{
            'lam':[0.0001, 0.005, 0.05],
            'lr':[0.01, 0.1]
        },
    }

    param_constraints_Lorenz = {
        'pwgc':{'taumax':[3, 5], 'alpha':[0.01, 0.05, 0.1]},
        'var':{'taumax':[3, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3]},      # 'var':{'taumax':[1, 2, 3, 4, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3]},
        'lgc':{'taumax':[3, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3], 'alphas':(1e-4, 5e-3, 1e-2, 2e-2, 5e-2)},      #'lgc':{'taumax':[1, 2, 3, 4, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3], 'alphas':(1e-4, 5e-3, 1e-2, 2e-2, 5e-2)},
        'pcmci':{'taumax':[3, 5], 'alpha':[0.01, 0.05, 0.1]},
        'varlingam':{'taumax':[3, 5], 'varlingamalpha':[0, 0.01, 0.05, 0.1, 0.3]},
        'dynotears':{'taumax':[3, 5], 'wthre':[0.01, 0.05, 0.1, 0.3],
                     'lambda_a':[0.001, 0.01, 0.1], 'lambda_w':[0.001, 0.005, 0.01]},
        'tsci':{'theta':[0.4, 0.5, 0.6], 'fnn_tol':[0.005, 0.01]},
        'cuts':{'input_step':[1, 3, 5, 10], 'batchsize':[32, 128], 'weight_decay':[0, 0.001, 0.003]},
        'cutsplus':{'input_step':[1, 3, 5, 10], 'batchsize':[32, 128], 'weight_decay':[0, 0.001, 0.003]},

        'ntsnotears':{
            'taumax':[1],
            'wthre':[0.1, 0.3, 0.5],
            'lambda_1':[(0.001, 0.1)],
            'lambda_2':[0.01]
        },

        'cmlp':{
            'lam':[0.0001, 0.005, 0.05],
            'lr':[0.0005, 0.001]
        },

        'clstm':{
            'lam':[0.0001, 0.005, 0.05],
            'lr':[0.0005, 0.001]
        },
    }

    METHODS = ['var', 'lgc', 'varlingam', 'pcmci', 'dynotears', 'ntsnotears', 'tsci', 'cmlp', 'clstm', 'cuts',
               'cutsplus']

    # Separate scenarios for VAR and Lorenz due to different noise_std
    SCENARIOS_VAR = [
        # Part I (4 columns): Vanilla, Mixed data, Trend and seasonality, Min-max normalization
        'vanilla', 'mixed_data_ratio0.5', 'trendseason', 'standardized_minmax',
        # Part II (5 columns): Latent confounders, Measurement error, Standardized, Missing, Nonstationary
        'confounder_rho0.5', 'measurement_error_gamma1.2', 'standardized_zscore', 'missing_prob0.4',
        'nonstationary_noisestd1.0_mean1.0'
    ]


    # Lorenz F=10 scenarios (noise_std=2.0, mean=2.5)
    SCENARIOS_LORENZ_F10 = [
        # Part I (4 columns): Vanilla, Mixed data, Trend and seasonality, Min-max normalization
        'vanilla', 'mixed_data_ratio0.5', 'trendseason', 'standardized_minmax',
        # Part II (5 columns): Latent confounders, Measurement error, Standardized, Missing, Nonstationary
        'confounder_rho0.5', 'measurement_error_gamma1.2', 'standardized_zscore', 'missing_prob0.4',
        'nonstationary_noisestd2.0_mean2.5'
    ]

    # Lorenz F=40 scenarios (noise_std=2.0, mean=3.5)
    SCENARIOS_LORENZ_F40 = [
        # Part I (4 columns): Vanilla, Mixed data, Trend and seasonality, Min-max normalization
        'vanilla', 'mixed_data_ratio0.5', 'trendseason', 'standardized_minmax',
        # Part II (5 columns): Latent confounders, Measurement error, Standardized, Missing, Nonstationary
        'confounder_rho0.5', 'measurement_error_gamma1.2', 'standardized_zscore', 'missing_prob0.4',
        'nonstationary_noisestd2.0_mean3.5'
    ]

    for p in [15]:
        for T in [500, 1000]:
            generate_latex_tables(
                data_model='VAR', p=p, T=T,
                methods=METHODS, scenarios=SCENARIOS_VAR,
                method_param_constraints=param_constraints_VAR,
                output_file=f'table_VAR_p{p}_T{T}.tex'
            )

    for p in [15]:
        for T in [500, 1000]:
            # Lorenz F=10
            generate_latex_tables(
                data_model='Lorenz', p=p, T=T, F=10,
                methods=METHODS, scenarios=SCENARIOS_LORENZ_F10,
                method_param_constraints=param_constraints_Lorenz,
                output_file=f'table_Lorenz_p{p}_T{T}_F10.tex'
            )

            # Lorenz F=40
            generate_latex_tables(
                data_model='Lorenz', p=p, T=T, F=40,
                methods=METHODS, scenarios=SCENARIOS_LORENZ_F40,
                method_param_constraints=param_constraints_Lorenz,
                output_file=f'table_Lorenz_p{p}_T{T}_F40.tex'
            )