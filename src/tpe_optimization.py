import pandas as pd
import numpy as np
import optuna
import os
from typing import List, Dict, Union, Tuple, Any
import argparse
from constants import FIXED_THREE_RATIOS, FIXED_TWO_RATIOS, THRESHOLDED_EXPECTED_ERROR, THRESHOLDED_WEIGHTED_BUDGET

def get_best_params_from_csv(data_csv_path: str, objective_col: str = "objective_value") -> Dict:
    """
    Reads a CSV file containing optimization data and returns the parameters
    corresponding to the best (minimum) objective value.

    Args:
        data_csv_path (str): The path to the CSV file.
        objective_col (str): The name of the objective value column.

    Returns:
        dict: A dictionary of the best parameters found in the CSV.
    """
    if not os.path.exists(data_csv_path):
        print(f"CSV file '{data_csv_path}' does not exist.")
        return {}

    df = pd.read_csv(data_csv_path)

    if objective_col not in df.columns:
        print(f"Objective column '{objective_col}' not found in CSV.")
        return {}
    
    if df.empty or df[objective_col].isna().all():
        print("CSV is empty or contains no completed evaluations.")
        return {}

    # Find the row with the minimum objective value
    best_row = df.loc[df[objective_col].idxmin()]

    # Convert to dictionary excluding the objective column
    best_params = best_row.drop(objective_col).to_dict()
    
    return best_params

def _load_optimization_data(
    data_csv_path: str, 
    space_definition: Dict[str, Tuple[str, Any]], 
    objective_col: str
) -> pd.DataFrame:
    """
    Loads optimization data from a CSV file. Initializes an empty DataFrame
    if the file doesn't exist or is empty, ensuring correct column dtypes.

    Args:
        data_csv_path (str): The path to the CSV file.
        space_definition (dict): The dictionary defining the hyperparameter space.
        objective_col (str): The name of the objective value column.

    Returns:
        pd.DataFrame: The loaded (or initialized) DataFrame with correct dtypes.
    """
    param_names = list(space_definition.keys())
    all_columns = param_names + [objective_col,"step"]

    if not os.path.exists(data_csv_path) or os.stat(data_csv_path).st_size == 0:
        df_empty = pd.DataFrame(columns=all_columns)
        for param, (kind, _) in space_definition.items():
            if kind == 'int':
                df_empty[param] = df_empty[param].astype(pd.Int64Dtype())
            elif kind == 'float':
                df_empty[param] = df_empty[param].astype(float)
        df_empty[objective_col] = df_empty[objective_col].astype(float)
        df_empty["step"] = df_empty["step"].astype(int)
        return df_empty

    df = pd.read_csv(data_csv_path)

    # Ensure correct data types
    for param, (kind, _) in space_definition.items():
        if param in df.columns:
            if kind == 'int':
                df[param] = pd.to_numeric(df[param], errors='coerce').astype(pd.Int64Dtype())
            elif kind == 'float':
                df[param] = pd.to_numeric(df[param], errors='coerce')
    
    if objective_col in df.columns:
        df[objective_col] = pd.to_numeric(df[objective_col], errors='coerce').astype(float)
            
    return df

# --- Main Optimization Function ---

def run_optimization_step(
    space_definition: Dict[str, Tuple[str, Any]], 
    data_csv_path: str, 
    num_initial_points: int = 30, 
    random_seed: int = 42
) -> Union[List[Dict], None]:
    """
    Performs one step of Bayesian Optimization using Optuna's TPE algorithm,
    managing state via a CSV file.

    Args:
        space_definition (dict): A dictionary defining the search space.
            Example: {'param_name': ('type', [values]), ...}
            Types can be 'int', 'float', 'categorical'.
            For 'float', you can add 'log' as a third element in the tuple for log scale.
        data_csv_path (str): The path to the CSV file for storing data.
        num_initial_points (int): The number of random trials before TPE starts.
        random_seed (int): Seed for reproducibility.

    Returns:
        list[dict] or None: A list of suggested parameter dictionaries, or None if
        there are pending evaluations that require manual intervention.
    """
    param_names = list(space_definition.keys())
    objective_col_name = 'objective_value'
    
    # Load existing data
    current_data = _load_optimization_data(data_csv_path, space_definition, objective_col_name)

    # Check for pending evaluations
    if current_data[objective_col_name].isna().any():
        print("\n--- ATTENTION REQUIRED ---")
        print("There are pending evaluations in your CSV.")
        print(f"Please fill in the '{objective_col_name}' column in '{data_csv_path}'.")
        return None
    # Configure the TPE sampler and create an Optuna study
    sampler = optuna.samplers.TPESampler(seed=random_seed, n_startup_trials=num_initial_points, constant_liar=True)
    #sampler = optuna.samplers.GPSampler(seed=random_seed, n_startup_trials=num_initial_points)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    # Define the search space for Optuna from the space_definition dict
    distributions = {}
    for name, (kind, args) in space_definition.items():
        if kind == 'float':
            log = len(args) > 2 and args[2] == 'log'
            distributions[name] = optuna.distributions.FloatDistribution(low=args[0], high=args[1], log=log)
        elif kind == 'int':
            distributions[name] = optuna.distributions.IntDistribution(low=args[0], high=args[1])
        elif kind == 'categorical':
            distributions[name] = optuna.distributions.CategoricalDistribution(choices=args[0])

    # "Tell" the study about all previously observed trials from the CSV
    if not current_data.empty:
        for _, row in current_data.iterrows():
            params = row[param_names].to_dict()
            value = row[objective_col_name]
            trial = optuna.trial.create_trial(
                params=params,
                value=value,
                distributions=distributions,
                state=optuna.trial.TrialState.COMPLETE
            )
            study.add_trial(trial)
    
    print(f"Told Optuna's study about {len(current_data)} past completed evaluations.")

    # Decide how many new points to ask for
    num_completed = len(study.trials)
    if num_completed < num_initial_points:
        num_to_ask = num_initial_points - num_completed
        print(f"Asking for {num_to_ask} initial random suggestions.")
    else:
        num_to_ask = 10  # Ask for 3 new TPE-optimized points
        print(f"Asking for {num_to_ask} optimized suggestions using TPE.")

    # "Ask" the study for new parameters
    suggested_params_list = []
    for _ in range(num_to_ask):
        trial = study.ask(fixed_distributions=distributions)
        suggested_params_list.append(trial.params)

    # Append new suggestions to the CSV for manual evaluation
    new_suggestions_df = pd.DataFrame(suggested_params_list)
    new_suggestions_df[objective_col_name] = np.nan
    new_suggestions_df["step"] = current_data["step"].max() + 1 if not current_data.empty else 0

    updated_data = pd.concat([current_data, new_suggestions_df], ignore_index=True)
    updated_data.to_csv(data_csv_path, index=False)

    print(f"\nAppended {len(suggested_params_list)} new suggestions to '{data_csv_path}'.")
    print("Please evaluate them and fill in the 'objective_value' column.")
    
    return suggested_params_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Bayesian Optimization with Optuna using TPE.")
    parser.add_argument("--method", type=str, required=True, 
                        choices=[THRESHOLDED_EXPECTED_ERROR, THRESHOLDED_WEIGHTED_BUDGET, 
                                 FIXED_THREE_RATIOS, FIXED_TWO_RATIOS],
                        help="The method for trust region construction.")
    
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to the CSV file where optimization results will be stored.")
    args = parser.parse_args()
    # Getting the model parameters using bayesian optimization
    epsilon = 1e-9  
    if args.method == THRESHOLDED_EXPECTED_ERROR:
        parameter_space = {
            "threshold": ("float", [0.5 + epsilon, 1.0 - epsilon]),
            "alpha": ("float", [-1.0 + epsilon, 1.0 - epsilon]),
        }
    elif args.method == THRESHOLDED_WEIGHTED_BUDGET:
        parameter_space = {
            "threshold": ("float", [0.5 + epsilon, 1.0 - epsilon]),
            "budget_ratio": ("float", [0.0 + epsilon, 1.0 - epsilon]),
        }
    elif args.method == FIXED_THREE_RATIOS:
        parameter_space = {
            "k_ratio": ("float", [0.0 + epsilon, 1.0 - epsilon]),
            "value_0_ratio": ("float", [0.0 + epsilon, 1.0 - epsilon]),
            "Delta_ratio": ("float", [0.0 + epsilon, 1.0 - epsilon]),
        }
    elif args.method == FIXED_TWO_RATIOS:
        parameter_space = {
            "k_ratio": ("float", [0.0 + epsilon, 1.0 - epsilon]),
            "Delta_ratio": ("float", [0.0 + epsilon, 1.0 - epsilon]),
        }

    
    parameters = run_optimization_step(space_definition=parameter_space,data_csv_path= args.results_path)
    if parameters is None:
        print("No suggested parameters found. Exiting.")
    else:
        print("Suggested parameters:")
        for param in parameters:
            print(param)