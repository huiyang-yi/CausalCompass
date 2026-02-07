import os
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import re

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
    """Parse parameters from filename using method-specific patterns."""
    param_str = filename.replace(f"{method}_", "").replace(".json", "")
    params = {}
    param_names = METHOD_PARAM_PATTERNS.get(method, [])

    for param_name in param_names:
        pattern = rf'(?:^|_){re.escape(param_name)}_?([0-9.eE+-]+(?:_[0-9.eE+-]+)*)'
        match = re.search(pattern, param_str)

        if match:
            value_str = match.group(1)

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

            elif '_' in value_str and param_name in ['lambda_1', 'lambda_2']:
                if param_name == 'lambda_1' and value_str.count('_') >= 1:
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
                        params[param_name] = tuple(values) if len(values) > 1 else values[0]
                else:
                    actual_value = value_str.split('_')[-1]
                    try:
                        if '.' in actual_value or 'e' in actual_value.lower():
                            params[param_name] = float(actual_value)
                        else:
                            params[param_name] = int(actual_value)
                    except ValueError:
                        continue
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
    """Check if parameters satisfy constraints."""
    for param_name, constraint in constraints.items():
        if param_name not in params:
            return False

        param_value = params[param_name]

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

        elif isinstance(constraint, tuple):
            if len(constraint) == 2:
                if all(isinstance(x, (int, float)) for x in constraint):
                    min_val, max_val = constraint
                    if not (min_val <= param_value <= max_val):
                        return False
                else:
                    if param_value != constraint:
                        return False
            else:
                if param_value != constraint:
                    return False

        elif isinstance(constraint, list):
            if param_value not in constraint:
                return False

        else:
            if param_value != constraint:
                return False

    return True


def load_best_result(result_dir, method, param_constraints=None):
    """Load the best result (highest AUPRC) for a given method."""
    if not os.path.exists(result_dir):
        return None

    pattern = os.path.join(result_dir, f"{method}_*.json")
    files = glob.glob(pattern)

    if not files:
        return None

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

                if auprc > best_auprc:
                    best_auprc = auprc
                    best_result = {
                        'auroc_mean':auroc,
                        'auprc_mean':auprc,
                    }
                elif auprc == best_auprc and best_result is not None:
                    if auroc > best_result['auroc_mean']:
                        best_auprc = auprc
                        best_result = {
                            'auroc_mean':auroc,
                            'auprc_mean':auprc,
                        }
        except (KeyError, json.JSONDecodeError):
            continue

    return best_result


def generate_origin_radar_table(data_model, p, T, F=None, methods=None, scenarios=None,
                                method_param_constraints=None, base_dir='results/synthetic',
                                output_file=None):
    """Generate Origin-compatible table for radar chart with DL aggregation."""

    if methods is None:
        methods = ['cutsplus', 'cuts', 'clstm', 'cmlp', 'var', 'lgc',
                   'varlingam', 'pcmci', 'dynotears', 'ntsnotears', 'tsci']

    # Deep learning methods for aggregation
    dl_methods = ['cutsplus', 'cuts', 'clstm', 'cmlp']

    display_methods = ['dl_best', 'var', 'lgc', 'varlingam', 'pcmci',
                       'dynotears', 'ntsnotears', 'tsci']

    # Default scenarios
    if scenarios is None:
        scenarios = [
            'vanilla', 'mixed_data_ratio0.5', 'trendseason', 'standardized_minmax',
            'confounder_rho0.5', 'measurement_error_gamma1.2', 'standardized_zscore',
            'missing_prob0.4', 'nonstationary_noisestd1.0_mean1.0'
        ]

    # Scenario display names
    scenario_labels = {
        'vanilla':'Vanilla',
        'mixed_data_ratio0.5':'Mixed_data',
        'trendseason':'Trend_and_seasonality',
        'standardized_minmax':'Min-max_normalization',
        'confounder_rho0.5':'Confounders',
        'measurement_error_gamma1.2':'Measurement_error',
        'standardized_zscore':'Standardized',
        'missing_prob0.4':'Missing',
        'nonstationary_noisestd1.0_mean1.0':'Nonstationary',
        'nonstationary_noisestd2.0_mean2.5':'Nonstationary',
        'nonstationary_noisestd2.0_mean3.5':'Nonstationary',
    }

    # Method display names
    method_labels = {
        'cutsplus':'CUTS+',
        'cuts':'CUTS',
        'clstm':'cLSTM',
        'cmlp':'cMLP',
        'var':'VAR',
        'lgc':'LGC',
        'varlingam':'VARLINGAM',
        'pcmci':'PCMCI',
        'dynotears':'DYNOTEARS',
        'ntsnotears':'NTS-NOTEARS',
        'tsci':'TSCI',
        'dl_best':'Deep learning-based'
    }

    # Collect results
    results_auroc = {}
    results_auprc = {}

    for scenario in scenarios:
        results_auroc[scenario] = {}
        results_auprc[scenario] = {}

        # Construct directory path
        if data_model == 'Lorenz' and scenario == 'mechanism_violation':
            dim_str = f"p{p}_T{T}"
        else:
            dim_str = f"p{p}_T{T}"
            if data_model == 'Lorenz' and F is not None:
                dim_str += f"_F{F}"

        result_dir = os.path.join(base_dir, data_model, scenario, dim_str)

        for method in methods:
            param_constraints = None
            if method_param_constraints is not None and method in method_param_constraints:
                param_constraints = method_param_constraints[method]

            result = load_best_result(result_dir, method, param_constraints)

            if result is not None:
                results_auroc[scenario][method] = result['auroc_mean']
                results_auprc[scenario][method] = result['auprc_mean']
            else:
                results_auroc[scenario][method] = None
                results_auprc[scenario][method] = None

        dl_auroc_values = [results_auroc[scenario][m] for m in dl_methods
                           if results_auroc[scenario].get(m) is not None]
        dl_auprc_values = [results_auprc[scenario][m] for m in dl_methods
                           if results_auprc[scenario].get(m) is not None]

        results_auroc[scenario]['dl_best'] = max(dl_auroc_values) if dl_auroc_values else None
        results_auprc[scenario]['dl_best'] = max(dl_auprc_values) if dl_auprc_values else None

    df_auroc = pd.DataFrame(index=[scenario_labels.get(s, s) for s in scenarios],
                            columns=[method_labels.get(m, m) for m in display_methods])

    df_auprc = pd.DataFrame(index=[scenario_labels.get(s, s) for s in scenarios],
                            columns=[method_labels.get(m, m) for m in display_methods])

    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(display_methods):
            auroc_val = results_auroc[scenario][method]
            auprc_val = results_auprc[scenario][method]

            # Truncate to 1 decimal place (not round) - same as LaTeX
            if auroc_val is not None:
                auroc_pct = auroc_val * 100
                auroc_truncated = int(auroc_pct * 10) / 10
                df_auroc.iloc[i, j] = f"{auroc_truncated:.1f}"
            else:
                df_auroc.iloc[i, j] = ""

            if auprc_val is not None:
                auprc_pct = auprc_val * 100
                auprc_truncated = int(auprc_pct * 10) / 10
                df_auprc.iloc[i, j] = f"{auprc_truncated:.1f}"
            else:
                df_auprc.iloc[i, j] = ""

    # Generate output filename
    if output_file is None:
        if data_model == 'VAR':
            output_base = f"origin_radar_{data_model}_p{p}_T{T}"
        elif data_model == 'Lorenz' and F is not None:
            output_base = f"origin_radar_{data_model}_p{p}_T{T}_F{F}"
        else:
            output_base = f"origin_radar_{data_model}_p{p}_T{T}"

    # Save files
    auroc_file = f"{output_base}_AUROC.txt"
    auprc_file = f"{output_base}_AUPRC.txt"

    df_auroc.to_csv(auroc_file, sep='\t', index=True)
    df_auprc.to_csv(auprc_file, sep='\t', index=True)

    print(f"✓ AUROC table saved to: {auroc_file}")
    print(f"✓ AUPRC table saved to: {auprc_file}")
    print(f"  (Deep learning-based = max of CUTS+, CUTS, cLSTM, cMLP)")

    return df_auroc, df_auprc


# Example usage
if __name__ == "__main__":

    param_constraints_VAR = {
        'var':{'taumax':[3, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3]},
        'lgc':{'taumax':[3, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3], 'alphas':(1e-4, 5e-3, 1e-2, 2e-2, 5e-2)},
        'pcmci':{'taumax':[3, 5], 'alpha':[0.01, 0.05, 0.1]},
        'varlingam':{'taumax':[3, 5], 'varlingamalpha':[0, 0.01, 0.05, 0.1, 0.3]},
        'dynotears':{'taumax':[3, 5], 'wthre':[0.01, 0.05, 0.1, 0.3],
                     'lambda_a':[0.001, 0.01, 0.1], 'lambda_w':[0.001, 0.005, 0.01]},
        'tsci':{'theta':[0.4, 0.5, 0.6], 'fnn_tol':[0.005, 0.01]},
        'cuts':{'input_step':[1, 3, 5, 10], 'batchsize':[32, 128], 'weight_decay':[0, 0.001, 0.003]},
        'cutsplus':{'input_step':[1, 3, 5, 10], 'batchsize':[32, 128], 'weight_decay':[0, 0.001, 0.003]},
        'ntsnotears':{'taumax':[3], 'wthre':[0.01, 0.05, 0.1, 0.3, 0.5],
                      'lambda_1':[0.001, 0.01, 0.1], 'lambda_2':[0.001, 0.005, 0.01]},
        'cmlp':{'lam':[0.0001, 0.005, 0.05], 'lr':[0.01, 0.1]},
        'clstm':{'lam':[0.0001, 0.005, 0.05], 'lr':[0.01, 0.1]},
    }

    param_constraints_Lorenz = {
        'var':{'taumax':[3, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3]},
        'lgc':{'taumax':[3, 5], 'threshold':[0, 0.01, 0.05, 0.1, 0.3], 'alphas':(1e-4, 5e-3, 1e-2, 2e-2, 5e-2)},
        'pcmci':{'taumax':[3, 5], 'alpha':[0.01, 0.05, 0.1]},
        'varlingam':{'taumax':[3, 5], 'varlingamalpha':[0, 0.01, 0.05, 0.1, 0.3]},
        'dynotears':{'taumax':[3, 5], 'wthre':[0.01, 0.05, 0.1, 0.3],
                     'lambda_a':[0.001, 0.01, 0.1], 'lambda_w':[0.001, 0.005, 0.01]},
        'tsci':{'theta':[0.4, 0.5, 0.6], 'fnn_tol':[0.005, 0.01]},
        'cuts':{'input_step':[1, 3, 5, 10], 'batchsize':[32, 128], 'weight_decay':[0, 0.001, 0.003]},
        'cutsplus':{'input_step':[1, 3, 5, 10], 'batchsize':[32, 128], 'weight_decay':[0, 0.001, 0.003]},
        'ntsnotears':{'taumax':[1], 'wthre':[0.1, 0.3, 0.5],
                      'lambda_1':[(0.001, 0.1)], 'lambda_2':[0.01]},
        'cmlp':{'lam':[0.0001, 0.005, 0.05], 'lr':[0.0005, 0.001]},
        'clstm':{'lam':[0.0001, 0.005, 0.05], 'lr':[0.0005, 0.001]},
    }

    METHODS = ['cutsplus', 'cuts', 'clstm', 'cmlp', 'var', 'lgc',
               'varlingam', 'pcmci', 'dynotears', 'ntsnotears', 'tsci']

    # VAR scenarios
    SCENARIOS_VAR = [
        'vanilla', 'mixed_data_ratio0.5', 'trendseason', 'standardized_minmax',
        'confounder_rho0.5', 'measurement_error_gamma1.2', 'standardized_zscore',
        'missing_prob0.4', 'nonstationary_noisestd1.0_mean1.0'
    ]

    # Lorenz F=10 scenarios
    SCENARIOS_LORENZ_F10 = [
        'vanilla', 'mixed_data_ratio0.5', 'trendseason', 'standardized_minmax',
        'confounder_rho0.5', 'measurement_error_gamma1.2', 'standardized_zscore',
        'missing_prob0.4', 'nonstationary_noisestd2.0_mean2.5'
    ]

    # Lorenz F=40 scenarios
    SCENARIOS_LORENZ_F40 = [
        'vanilla', 'mixed_data_ratio0.5', 'trendseason', 'standardized_minmax',
        'confounder_rho0.5', 'measurement_error_gamma1.2', 'standardized_zscore',
        'missing_prob0.4', 'nonstationary_noisestd2.0_mean3.5'
    ]

    print("=" * 60)
    print("Generating Origin Radar Chart Tables")
    print("=" * 60)

    # Generate all combinations
    for p in [15]:
        for T in [500, 1000]:
            # VAR
            print(f"\n--- VAR: p={p}, T={T} ---")
            generate_origin_radar_table(
                data_model='VAR', p=p, T=T,
                methods=METHODS, scenarios=SCENARIOS_VAR,
                method_param_constraints=param_constraints_VAR
            )

            # Lorenz F=10
            print(f"\n--- Lorenz F=10: p={p}, T={T} ---")
            generate_origin_radar_table(
                data_model='Lorenz', p=p, T=T, F=10,
                methods=METHODS, scenarios=SCENARIOS_LORENZ_F10,
                method_param_constraints=param_constraints_Lorenz
            )

            # Lorenz F=40
            print(f"\n--- Lorenz F=40: p={p}, T={T} ---")
            generate_origin_radar_table(
                data_model='Lorenz', p=p, T=T, F=40,
                methods=METHODS, scenarios=SCENARIOS_LORENZ_F40,
                method_param_constraints=param_constraints_Lorenz
            )

    print("\n" + "=" * 60)
    print("All tables generated successfully!")
    print("=" * 60)