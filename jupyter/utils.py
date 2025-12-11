import re
import pandas as pd
import gurobi_logtools as glt
from pathlib import Path
import os
def get_trust_region_generation_timeout_instances(method_path):
    trust_region_construction_log = method_path.parent / f"{method_path.name}_construction_log.txt"
    if not trust_region_construction_log.exists():
        print(f"Warning: No trust region construction log found for method {method_path}.")
        return []
    with open(trust_region_construction_log, 'r') as file:
        lines = file.readlines()
    timeout_instances = set()
    for line in lines:
        if "Timeout:" in line:
            instance_path = Path(line.split(" ")[-1].strip())
            timeout_instances.add(instance_path.name)
    return timeout_instances

def check_is_feasible_and_without_error(file_path):
    status = "ERROR"
    with_best_objective = False
    orignal_folder_name = Path(file_path).parent.name.replace("_log","")
    construction_log = Path(file_path).parent.parent / f"{orignal_folder_name}_construction_log.txt"
    if construction_log.exists():
        with open(construction_log, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "Timeout:" in line:
                    instance_name = Path(line.split(" ")[-1].strip()).name
                    if instance_name == Path(file_path).name.replace(".log", ""):
                        print(f"Warning: This doesnt have a logfile because the trust region was not constructed due to timeout {file_path}")
                        return "TIMEOUT"
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Model is infeasible" in line:
                return "INFEASIBLE"
            elif "Error" in line:
                if "Error 10001: Out of memory" in line:
                    return "OUT_OF_MEMORY"
                return "ERROR"
            elif "Best objective" in line:
                with_best_objective = True
    if with_best_objective:
        status = "OK"
    else:
        print("Warning: The log file does not contain a best objective line. It will be considered an file with an error.",file_path)
    return status

def get_incumbents(file_path):
    results = glt.parse(str(file_path))
    nodelog_progress = results.progress("nodelog")
    df = nodelog_progress[["Incumbent","Time"]].copy()
    df.rename(columns={"Incumbent": "incumbent", "Time": "time"}, inplace=True)
    if df.isnull().values.any():
        df = df.dropna()
    return df

def get_logs_with_error(instance_log_dict):
    error_logs = {}
    for instance, log_paths in instance_log_dict.items():
        for method, log_file_path in log_paths.items():
            status = check_is_feasible_and_without_error(log_file_path)
            if status == "ERROR":
                error_logs[(instance, method)] = log_file_path
    return error_logs

def calculate_primal_gap(incumbent, bks):
    if bks == 0 and incumbent == 0:
        return 0.0
    elif bks * incumbent < 0:
        return 1.0
    else:
        denominator = max(abs(incumbent), abs(bks))
        return abs(incumbent - bks) / denominator
    
def compute_primal_integral(df,t_max):
    df_primal_gap = df[["time","primal_gap"]].copy()
    #add time 0 primal gap = 1
    df_primal_gap = pd.concat([pd.DataFrame({"time":[0],"primal_gap":[1]}),df_primal_gap])
    #add time = t_max primal gap = last primal gap
    df_primal_gap = pd.concat([df_primal_gap,pd.DataFrame({"time":[t_max],"primal_gap":[df_primal_gap["primal_gap"].iloc[-1]]})])
    #compute delta time
    df_primal_gap["delta_time"] = df_primal_gap["time"].diff().shift(-1)
    #discard last row
    df_primal_gap = df_primal_gap.iloc[:-1]
    #return df_primal_gap
    return (df_primal_gap["primal_gap"]*df_primal_gap["delta_time"]).sum()

def get_all_logs_for_instance(instance_log_dict, objective):
    # ex instance_log_dict = {
    #     "instance_1": {
    #         "method_a_b_c_log_path": "/path/to/log_a.log",
    #         "method_d_e_f_log_path": "/path/to/log_d.log",
    #     },
    #     "instance_2": {
    #         "method_a_b_c_log_path": "/path/to/log_a.log",
    #         "method_d_e_f_log_path": "/path/to/log_d.log",
    #     },
    #     "instance_3": {
    #         "method_a_b_c_log_path": "/path/to/log_a.log",
    #         "method_d_e_f_log_path": "/path/to/log_d.log",
    #     },
    df = pd.DataFrame()
    instances_with_timeout = set()
    for instance, log_paths in instance_log_dict.items():
        df_instance = pd.DataFrame()
        
        for method, log_file_path in log_paths.items():
            status = check_is_feasible_and_without_error(log_file_path)
            if status == "OK":
                incumbents = get_incumbents(log_file_path)
                if incumbents.empty:
                    print(f"Warning: No incumbents found in log file {log_file_path}.")
                    continue
                incumbents["method"] = method
                df_instance = pd.concat([df_instance, incumbents], axis=0)
            elif status == "TIMEOUT":
                print(f"XX: log file {log_file_path} indicates that the trust region construction timed out. Skipping this log.")
                instances_with_timeout.add(instance)
                continue
            elif status == "INFEASIBLE":
                print(f"Warning: log file {log_file_path} indicates that the problem is infeasible. Skipping this log.")
                continue
            elif status == "OUT_OF_MEMORY":
                print(f"Warning: log file {log_file_path} indicates that the problem ran out of memory. Skipping this log.")
                continue
            elif status == "ERROR":
                raise Exception(f"Error in log file {log_file_path}. Please check the log for details.")
        if df_instance.empty:
            print(f"Warning: No feasible logs found for instance {instance}.")
            continue
        # Clean the incumbents (remove duplicates)
        if objective=="min":
            df_instance = df_instance.groupby(["method","time"])["incumbent"].min()
        elif objective=="max":
            df_instance = df_instance.groupby(["method","time"])["incumbent"].max()
        else:
            raise ValueError(f"Objective {objective} not recognized. Use 'min' or 'max'.")
        #create new dataframe with the time, max_best_bjective and method
        df_instance = df_instance.reset_index()
        df_instance = df_instance.sort_values("time")
        if objective=="min":
            best_known_solution = df_instance["incumbent"].min()
        else:
            best_known_solution = df_instance["incumbent"].max()
        df_instance["primal_gap"] = df_instance.apply(lambda row: calculate_primal_gap(row["incumbent"], best_known_solution), axis=1)
        df_instance["instance"] = instance
        df = pd.concat([df, df_instance], axis=0)
    #print(df.columns)
    func_primal_integral = lambda x: compute_primal_integral(x,df["time"].max())
    df_primal_integral = df.groupby(["instance", "method"], group_keys=False)[["time","primal_gap"]].apply(func_primal_integral).reset_index()
    df_primal_integral.columns = ["instance", "method","primal_integral"]
    #foreach instance, method in instance_log_dict compute primal_integral as
    # if df_computed_primal_integral does not contain the instance, method, because there is no incumbents then
    # primal_integral = df["time"].max() * 1.0
    # else:
    # primal_integral = df_computed_primal_integral["primal_integral"]
    for instance, log_paths in instance_log_dict.items():
        if instance in instances_with_timeout:
            print(f"YY: Instance {instance} had timeout during trust region construction. Ignoring instance completely")
            #remove all rows with this instance from df_primal_integral
            df_primal_integral = df_primal_integral[df_primal_integral["instance"] != instance]
            continue
        for method, log_file_path in log_paths.items():
            query = (df_primal_integral["instance"] == instance) & (df_primal_integral["method"] == method)
            if df_primal_integral[query].empty:
                print(f"Warning: No incumbents found for instance {instance} and method {method}. Setting primal_integral to df['time'].max() * 1.0.")
                df_primal_integral = pd.concat([df_primal_integral, pd.DataFrame({
                    "instance": [instance], 
                    "method": [method], 
                    "primal_integral": [df["time"].max() * 1.0]
                })], axis=0)
    #count unique instances by method
    number_of_instances_per_method = df_primal_integral.groupby("method")["instance"].nunique().reset_index()
    #assert that each method has exactly len(instance_log_dict) different instances
    #get number of first row
    first_row_count = number_of_instances_per_method["instance"].iloc[0]

    assert (number_of_instances_per_method["instance"] == first_row_count).all(), \
        f"Not all methods have the same number of instances. {number_of_instances_per_method}"
    return df_primal_integral,df



def construct_instance_log_dict(baseline_path, methods_paths):
    #it is assumed that the baseline_path contains all the instances
    baseline_path = Path(baseline_path)
    baseline_log_path = baseline_path.parent / f"{baseline_path.name}_log"
    if not baseline_path.is_dir():
        raise ValueError(f"Baseline path {baseline_path} is not a directory.")
    if not methods_paths:
        raise ValueError("No methods paths provided.")
    methods_paths = [Path(method_path) for method_path in methods_paths]
    timeout_instances = {method_path: get_trust_region_generation_timeout_instances(method_path) for method_path in methods_paths}
    #prints length of timeout_instances
    for method_path, timeout_instances_list in timeout_instances.items():
        print(f"Method {method_path.name} has {len(timeout_instances_list)} instances with timeout.")
    methods_paths = [
        (method_path, method_path.parent / f"{method_path.name}_log") for method_path in methods_paths
    ]
    #for method_path, method_log_path in methods_paths:
    #    if not method_path.is_dir():
    #        raise ValueError(f"Method path {method_path} is not a directory.")
    #    if not method_log_path.is_dir():
    #        raise ValueError(f"Method log path {method_log_path} is not a directory.")
    #instances_names are the files inside the baseline_path directory
    instances_names = [entry.name for entry in baseline_path.iterdir() if entry.is_file()]
    instance_log_dict = {}
    for instance in instances_names:
        instance_log_dict[instance] = {}
        instance_log_dict[instance]["baseline"] = baseline_log_path / f"{instance}.log"
        for method_path, method_log_path in methods_paths:
            
            method_name = method_path.name
            instance_path = method_path / instance


            if instance_path.exists() or instance_path.name in timeout_instances[method_path]: #if trust region was created for this method or trust region had a timeout
                if instance_path.name in timeout_instances[method_path]:
                    print(f"Warning: Trust region generation for instance {instance} in method {method_name} had a timeout. It will be assumed that the log for this non existent file should exists (having the penalization of not solving the problem).")
                instance_log_dict[instance][method_name] = method_log_path / f"{instance}.log"
            else: #It is assumed that there is not a trust region file because the method choose to not generate any or equivalently, it choose to use the original file.
                print(f"Warning: No trust region file found for instance {instance} in method {method_name}. Using original log file instead.")
                instance_log_dict[instance][method_name] = instance_log_dict[instance]["baseline"]
        # check if at least one method has a valid log file
    return instance_log_dict

def create_temp_file_list(file_paths: list[Path], output_filename="temp_file_list.txt"):
    """
    Sorts a list of file paths (Path objects) by size (largest first) and writes them to a file,
    one path per line, to be compatible with bash's mapfile -t.

    Args:
        file_paths (list[Path]): A list of pathlib.Path objects, where each is a file path.
        output_filename (str): The name of the file to write the sorted list to.
    """
    file_sizes = []
    for path_obj in file_paths:
        try:
            # Get the size of the file using the Path object's stat() method
            # path_obj.stat().st_size or os.path.getsize(path_obj) both work
            size = path_obj.stat().st_size
            file_sizes.append((path_obj, size))
        except FileNotFoundError:
            print(f"Warning: File not found - {path_obj}. Skipping.")
        except Exception as e:
            print(f"Error getting size for {path_obj}: {e}. Skipping.")

    # Sort by size in descending order (largest first)
    # The 'key' for sorting is a lambda function that returns the second element of the tuple (the size)
    sorted_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)

    with open(output_filename, "w") as f:
        for path_obj, _ in sorted_files:
            # Convert the Path object to a string before writing to the file
            f.write(str(path_obj) + "\n")

    print(f"Sorted file list written to {output_filename}")

