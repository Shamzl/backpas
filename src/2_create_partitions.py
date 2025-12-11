import os
import pickle as pkl
import argparse

def main():
    parser = argparse.ArgumentParser(description="Creates a partition.pkl file. This file specificies which instances will be used as train, valid and test for machine learning and trust region experiments.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory containing instance and backbone directories.")
    parser.add_argument("--ml_dataset_path", type=str, required=False, help="Path to the machine learning dataset directory. This is used to ensure that the ml partitions (train, valid, test) has the corresponding ml_dataset files.")
    parser.add_argument("--wkdir_path", type=str, required=True, help="Path to the destination working directory where a folder for this dataset will be created.")

    parser.add_argument("--ml_train_prefix", type=str, default="train_")
    parser.add_argument("--ml_valid_prefix", type=str, default="valid_")
    parser.add_argument("--ml_test_prefix", type=str, default="test_")
    parser.add_argument("--trust_region_valid_prefix", type=str, default="trust_region_valid_")
    parser.add_argument("--trust_region_test_prefix", type=str, default="trust_region_test_")


    args = parser.parse_args()
    dataset_path = args.dataset_path
    dataset_name = os.path.basename(dataset_path)
    ml_dataset_path = args.ml_dataset_path
    
    instance_path=os.path.join(dataset_path,"instance")
    all_instances = os.listdir(instance_path)
    wkdir_path = args.wkdir_path

    dataset_wkdir = os.path.join(wkdir_path, dataset_name)
    os.makedirs(dataset_wkdir, exist_ok=True)

    ml_partitions = {
        "train": [f for f in all_instances if f.startswith(args.ml_train_prefix)],
        "valid": [f for f in all_instances if f.startswith(args.ml_valid_prefix)],
        "test": [f for f in all_instances if f.startswith(args.ml_test_prefix)]
    }

    if ml_dataset_path:
        ml_partitions = {
            "train": [f for f in ml_partitions["train"] if os.path.exists(os.path.join(ml_dataset_path, f + ".pkl"))],
            "valid": [f for f in ml_partitions["valid"] if os.path.exists(os.path.join(ml_dataset_path, f + ".pkl"))],
            "test": [f for f in ml_partitions["test"] if os.path.exists(os.path.join(ml_dataset_path, f + ".pkl"))]
        }
    trust_regions_partitions = {
        "valid": [f for f in all_instances if f.startswith(args.trust_region_valid_prefix)],
        "test": [f for f in all_instances if f.startswith(args.trust_region_test_prefix)]
    }

    # Check if there is intersection between train, valid and test partitions
    all_partitions = {
        "ml_train": ml_partitions["train"],
        "ml_valid": ml_partitions["valid"],
        "ml_test": ml_partitions["test"],
        "trust_region_valid": trust_regions_partitions["valid"],
        "trust_region_test": trust_regions_partitions["test"]
    }
    for partition_name_a, partition_files_a in all_partitions.items():
        for partition_name_b, partition_files_b in all_partitions.items():
            if partition_name_a == partition_name_b:
                continue
            if "valid" in partition_name_a and "valid" in partition_name_b:
                continue
            if "test" in partition_name_a and "test" in partition_name_b:
                continue
            intersection = set(partition_files_a).intersection(set(partition_files_b))
            if intersection:
                raise ValueError(f"There is an intersection between {partition_name_a} and {partition_name_b} partitions. Please check the prefixes used for partitioning.")

    readme_log = f"Dataset: {dataset_name}\n"
    readme_log += f"ML Dataset Path: {ml_dataset_path}\n"
    readme_log += f"Working Directory: {dataset_wkdir}\n"
    readme_log += f"ML Partitions:\n"
    readme_log += f"\tTrain: {len(ml_partitions['train'])}\n"
    readme_log += f"\tValid: {len(ml_partitions['valid'])}\n"
    readme_log += f"\tTest: {len(ml_partitions['test'])}\n"
    readme_log += f"Trust Regions Partitions:\n"
    readme_log += f"\tValid: {len(trust_regions_partitions['valid'])}\n"
    readme_log += f"\tTest: {len(trust_regions_partitions['test'])}\n"

    with open(f'{dataset_wkdir}/readme.txt', 'w') as readme:
        readme.write(readme_log)
    
    with open(f'{dataset_wkdir}/partitions.pkl', 'wb') as handle:
        pkl.dump({
            "ml_partitions": ml_partitions,
            "trust_regions_partitions": trust_regions_partitions
        }, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print(readme_log)
if __name__ == '__main__':
    main()