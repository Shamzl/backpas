import argparse
import torch
from constants import LITERALS_GRAPH, VARIABLES_GRAPH, THRESHOLDED_EXPECTED_ERROR, THRESHOLDED_WEIGHTED_BUDGET, FIXED_THREE_RATIOS, FIXED_TWO_RATIOS
from pathlib import Path
from GCN import BackbonePredictor
import pickle as pkl
from tpe_optimization import run_optimization_step, get_best_params_from_csv
from create_trust_region import ThresholdedExpectedErrorTrustRegionConstructor, ThresholdedWeightedBudgetTrustRegionConstructor, FixedThreeRatiosTrustRegionConstructor, FixedTwoRatiosTrustRegionConstructor, TrustRegionConstructor
import shutil
import multiprocessing
RUN_PURPOSE_VALIDATION = "validation"
RUN_PURPOSE_TEST = "test"
def parse_model_path(model_path: str):
    parts = model_path.split('_')
    graph_type = parts[2]
    num_layers = int(parts[3])
    layer_type = parts[4]
    use_literals_message = 'with_literals_message' in model_path
    
    return graph_type, num_layers, layer_type, use_literals_message
def main():
    parser = argparse.ArgumentParser(description="Create trust regions using a machine learning model.",formatter_class=argparse.RawTextHelpFormatter)

    # File sources
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory containing instance and backbone directories.")
    parser.add_argument("--dataset_wkdir_path", type=str, required=True, help="Path to the working directory for the dataset.")
    
    # Method
    parser.add_argument("--method", type=str, required=True, choices=[
        THRESHOLDED_EXPECTED_ERROR,
        THRESHOLDED_WEIGHTED_BUDGET,
        FIXED_THREE_RATIOS,
        FIXED_TWO_RATIOS
    ], help="The method to use for creating trust regions.")

    # Model parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the machine learning model weights. The containing folder of this file should be graph_with_{graph_type}_{num_layers}_{layer_type}[_with_literals_message].")
    parser.add_argument("--use_cuda", action="store_true", help="If set, use CUDA for training the model. If not set, use CPU.")

    # Running purpose
    parser.add_argument("--run_purpose", type=str, required=True, choices=[RUN_PURPOSE_VALIDATION, RUN_PURPOSE_TEST], help="The purpose of the run. If validation then it will perform the bayesian optimization of the parameters in the validation partition, if test it will use the parameters found in the validation partition to create the trust regions in the test partition.")

    args = parser.parse_args()
    # Parse the model path to extract graph type, number of layers, layer type, and whether literals message is used
    model_path = Path(args.model_path)
    graph_type, num_layers, layer_type, use_literals_message = parse_model_path(model_path.parent.name)

    # Check graph_type and use_literals_message compatibility
    if use_literals_message and graph_type != LITERALS_GRAPH:
        raise ValueError(f"Literal message passing can only be used with the '{LITERALS_GRAPH}' graph type.")
    # Check cuda availability
    if args.use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your PyTorch installation or use the CPU.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Read the partitions file
    partitions = pkl.load(open(Path(args.dataset_wkdir_path) / "partitions.pkl", "rb"))

    method_path = Path(args.dataset_wkdir_path) / RUN_PURPOSE_VALIDATION / model_path.parent.name / f"trust_region_{args.method}"
    if not method_path.exists():
        method_path.mkdir(parents=True, exist_ok=True)
    results_path = method_path / "results.csv"

    # Get the parameters based on the run purpose and select the instances to process
    if args.run_purpose == RUN_PURPOSE_VALIDATION:
        # Getting the model parameters using bayesian optimization
        epsilon = 1e-9  
        if args.method == THRESHOLDED_EXPECTED_ERROR:
            parameter_space = {
                "threshold": ("float", [0.5 + epsilon, 1.0 - epsilon]),
                "alpha": ("float", [-1.0, 1.0 - epsilon]),
            }
        elif args.method == THRESHOLDED_WEIGHTED_BUDGET:
            parameter_space = {
                "threshold": ("float", [0.5 + epsilon, 1.0 - epsilon]),
                "budget_ratio": ("float", [0.0 + epsilon, 1.0 - epsilon]),
            }
        elif args.method == FIXED_THREE_RATIOS:
            parameter_space = {
                "k_ratio": ("float", [0.0 + epsilon, 1.0]),
                "value_0_ratio": ("float", [0.0, 1.0]),
                "Delta_ratio": ("float", [0.0, 1.0 - epsilon]),
            }
        elif args.method == FIXED_TWO_RATIOS:
            parameter_space = {
                "k_ratio": ("float", [0.0 + epsilon, 1.0]),
                "Delta_ratio": ("float", [0.0, 1.0 - epsilon]),
            }

        
        parameters = run_optimization_step(space_definition=parameter_space,data_csv_path= results_path)

        if parameters is None:
            print("No suggested parameters found. Exiting.")
            return
        instances = partitions["trust_regions_partitions"]["valid"]
    elif args.run_purpose == RUN_PURPOSE_TEST:
        # Use the parameters found in the validation partition
        parameters = [get_best_params_from_csv(results_path)]
        # Change method_path to the test destination path
        method_path = Path(args.dataset_wkdir_path) / RUN_PURPOSE_TEST / model_path.parent.name / f"trust_region_{args.method}"
        instances = partitions["trust_regions_partitions"]["test"]
    
    if len(instances) == 0:
        raise ValueError(f"No instances found in the trust region partition (with run purpose {args.run_purpose}). Please check the partitions file.")
    
    # Load the model
    model = BackbonePredictor(
        graph_type=graph_type,
        num_layers=num_layers,
        layer_type=layer_type,
        use_literals_message=use_literals_message
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # Copy baseline instances
    baseline_path = Path(args.dataset_wkdir_path) / args.run_purpose / "baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)
    for instance in instances:
        instance_path_input = Path(args.dataset_path) / "instance" / instance
        instance_path_output = baseline_path / instance
        # Copy the instance file to the baseline path
        shutil.copy(instance_path_input, instance_path_output)
    # parameters is a list of dictionaries, each containing the different configurations.
    for params in parameters:
        # Update the arguments based on the suggested parameters
        if args.method == THRESHOLDED_EXPECTED_ERROR:
            config_name = f"threshold_{params['threshold']}_alpha_{params['alpha']}"
            log_file_path = method_path / f"{config_name}_construction_log.txt"
            trust_region_method = ThresholdedExpectedErrorTrustRegionConstructor(
                ml_model=model,
                graph_type=graph_type,
                threshold=params['threshold'],
                alpha=params['alpha'],
                log_file=log_file_path
            )

        elif args.method == THRESHOLDED_WEIGHTED_BUDGET:
            config_name = f"threshold_{params['threshold']}_budget_{params['budget_ratio']}"
            log_file_path = method_path / f"{config_name}_construction_log.txt"
            trust_region_method = ThresholdedWeightedBudgetTrustRegionConstructor(
                ml_model=model,
                graph_type=graph_type,
                threshold=params['threshold'],
                budget=params['budget_ratio'],
                log_file=log_file_path
            )
        elif args.method == FIXED_TWO_RATIOS:
            config_name = f"k_{params['k_ratio']}_delta_{params['Delta_ratio']}"
            log_file_path = method_path / f"{config_name}_construction_log.txt"
            trust_region_method = FixedTwoRatiosTrustRegionConstructor(
                ml_model=model,
                graph_type=graph_type,
                k_ratio=params['k_ratio'],
                Delta_ratio=params['Delta_ratio'],
                log_file=log_file_path
            )
        elif args.method == FIXED_THREE_RATIOS:
            config_name = f"k_{params['k_ratio']}_value_0_{params['value_0_ratio']}_delta_{params['Delta_ratio']}"
            log_file_path = method_path / f"{config_name}_construction_log.txt"
            trust_region_method = FixedThreeRatiosTrustRegionConstructor(
                ml_model=model,
                graph_type=graph_type,
                k_ratio=params['k_ratio'],
                value_0_ratio=params['value_0_ratio'],
                Delta_ratio=params['Delta_ratio'],
                log_file=log_file_path
            )
        trust_region_destination_path = method_path / config_name
        trust_region_destination_path.mkdir(parents=True, exist_ok=True)
        # Create the trust region instances
        for instance in instances:
            instance_path_input = Path(args.dataset_path) / "instance" / instance
            instance_path_output = trust_region_destination_path / instance
            trust_region_method.process_instance(instance_path_input, instance_path_output)


if __name__ == '__main__':
    #multiprocessing.set_start_method('spawn', force=True)  # Ensure compatibility with CUDA
    main()