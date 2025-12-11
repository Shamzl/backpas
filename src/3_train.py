import os
import pickle as pkl
import argparse
import torch
from GCN import BackbonePredictor, GraphBackboneDataset
import torch_geometric
import numpy as np
from torcheval.metrics.functional import multiclass_accuracy,multiclass_f1_score,multiclass_auroc, multiclass_auprc,multiclass_precision, multiclass_recall , retrieval_precision, multiclass_confusion_matrix
import torch.nn.functional as F
import time 
from constants import LITERALS_GRAPH, VARIABLES_GRAPH, GCN_LAYER, GTR_LAYER
import pandas as pd
import logging
logging.getLogger().setLevel(logging.ERROR)
class MetricAggregator():
    def __init__(self, aggregation_metrics_func_dict:dict, individual_metrics_func_dict:dict = None):
        """
            aggregation_metrics_func_dict: {name_of_the_metric: function_to_compute_the_metric}
            individual_metrics_func_dict: {general_name_of_the_metric: (function_to_compute_the_metric,{specific_name_of_the_metric:index_to_acces_the_metric})}
        """
        self.aggregation_metrics_func_dict = aggregation_metrics_func_dict
        self.individual_metrics_func_dict = individual_metrics_func_dict
        self.metric_names = [k for k in self.aggregation_metrics_func_dict]
        self.metric_names.sort()
        self.multi_metric_names = [k for k in self.individual_metrics_func_dict]
        self.multi_metric_names.sort()
        self.multi_metric_specific_names = []
        for general_name in self.multi_metric_names:
            for specific_name in self.individual_metrics_func_dict[general_name][1]:
                self.multi_metric_specific_names.append(general_name+"_"+specific_name)
        self.reset()
    def get_metric_names(self):
        return self.metric_names+self.multi_metric_specific_names
    def reset(self):
        self.values = [[] for _ in self.metric_names] + [[] for _ in self.multi_metric_specific_names]
    def update(self,pred,target):
        for index, metric in enumerate(self.metric_names):
            self.values[index].append(self.aggregation_metrics_func_dict[metric](pred,target).item())
        index = len(self.metric_names)
        for metric in self.multi_metric_names:
            metric_func, indexes_to_access_metric = self.individual_metrics_func_dict[metric]
            value = metric_func(pred,target)
            for index_to_access in indexes_to_access_metric.values():
                self.values[index].append(value[index_to_access].item())
                index+=1
    def aggregate(self):
        metrics = np.array(self.values)
        return metrics.mean(axis=1).tolist()
    
def create_precision_at_p(p:float):
    def precision_at_p(pred,target):
        #pred is a tensor with shape (n_variables,3)
        pred = F.softmax(pred,dim=1)
        #the last dimension is the probability of each category (backbone with value 0, backbone with value 1, non-backbone)
        #calculate max between backbone with value 0 and backbone with value 1
        predicted_is_backbone = pred[:,:2].max(dim=1)[0]
        #predicted_is_backbone = pred[:,0]+pred[:,1]
        #select first 2 columns of the prediction
        is_prediction_correct = (pred[:,:2].argmax(dim=1)==target).int()
        k = int(predicted_is_backbone.nelement()*p)
        #top_k_index = torch.topk(predicted_is_backbone,k)[1]
        #return is_prediction_correct[top_k_index].sum().float()/k
        return retrieval_precision(predicted_is_backbone,is_prediction_correct,k)
    return precision_at_p

def create_dataloader_from_names(names:list, dataset_path:str, batch_size:int, graph_type:str, shuffle:bool=True):
    files = [ os.path.join(dataset_path,name+".pkl") for name in names]
    graph_dataset = GraphBackboneDataset(files, graph_type)
    return torch_geometric.loader.DataLoader(graph_dataset, batch_size=batch_size, shuffle=shuffle)

def one_epoch(model, data_loader, metric_aggregator, device, batch_accumulation,optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    mean_loss = []
    metric_aggregator.reset()
    n_samples_processed = 0
    backward_passes = 0
    n_train_samples = len(data_loader)
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            batch = batch.to(device)
            pred = model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )

            start_index = 0
            loss = torch.zeros(1,device=device)
            for i in range(batch.num_original_bin_variables.shape[0]):
                end_index = start_index+batch.num_original_bin_variables[i]
                # instance slices
                instance_pred = pred[start_index:end_index]
                instance_target = batch.target[start_index:end_index]
                instance_loss = F.cross_entropy(instance_pred,instance_target,reduction="mean")
                loss += instance_loss
                # stats
                mean_loss.append(instance_loss.item())
                metric_aggregator.update(instance_pred,instance_target)
                start_index = end_index
            n_samples_processed += batch.num_graphs
            if optimizer is not None:
                (loss/batch.num_graphs).backward()
                if (n_samples_processed >= (backward_passes+1) * batch_accumulation) or n_samples_processed==n_train_samples:
                    optimizer.step()
                    optimizer.zero_grad()
                    backward_passes+=1
            assert end_index == pred.shape[0], f"End index {end_index} does not match prediction shape {pred.shape[0]} for step {step}."
    return np.mean(mean_loss), metric_aggregator.aggregate()
def full_train(model,optimizer,epochs,best_model_path,last_model_path,last_optimizer_path,add_to_log, metric_aggregator, device,batch_accumulation, train_loader,valid_loader=None,starting_epoch=0,starting_best_loss=float("inf")):
    best_loss = starting_best_loss
    for epoch in range(starting_epoch, epochs):
        print(epoch)
        begin=time.time()
        loss, metrics = one_epoch(model, train_loader, metric_aggregator, device,batch_accumulation, optimizer=optimizer)
        epoch_time = time.time()-begin
        add_to_log(epoch,epoch_time,"train",[loss]+metrics)
        begin=time.time()
        if valid_loader:
            begin=time.time()
            loss, metrics = one_epoch(model, valid_loader, metric_aggregator, device, batch_accumulation, optimizer=None)
            epoch_time = time.time()-begin
            add_to_log(epoch,epoch_time,"valid",[loss]+metrics)
            begin=time.time()
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(),best_model_path)
        torch.save(model.state_dict(), last_model_path)
        torch.save(optimizer.state_dict(), last_optimizer_path)

def main():
    parser = argparse.ArgumentParser(description="Train a machine learning model on a dataset.")
    # File sources
    parser.add_argument("--ml_dataset_path", type=str, required=True, help="Path to the machine learning dataset directory.")
    parser.add_argument("--dataset_wkdir_path", type=str, required=True, help="Path to the working directory for the dataset.")
    
    # Model parameters
    parser.add_argument("--graph_type", type=str, choices=[LITERALS_GRAPH, VARIABLES_GRAPH], default=LITERALS_GRAPH, help="Type of bipartite graphs contained in ml_dataset_path.")
    parser.add_argument("--layer_type", type=str, choices=[GCN_LAYER, GTR_LAYER], default=GTR_LAYER, help=f"Type of GNN layer to use in the model. {GCN_LAYER} for Graph Convolutional Network, {GTR_LAYER} for Graph Transformer.")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of GNN layers in the model.")
    parser.add_argument("--use_literals_message", action="store_true", help="Whether to use pass messages between literals after each GNN layer. If not set, only messages between constraints and variables will be performed. This can only be used with the literals graph type.")
    
    # Running purpose
    parser.add_argument("--only_run_test", action="store_true", help="If set, only run the test phase without training the model.")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training the model.")
    parser.add_argument("--batch_accumulation_size", type=int, default=32, help="Number of samples to accumulate gradients before performing an optimizer step.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train the model.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--use_cuda", action="store_true", help="If set, use CUDA for training the model. If not set, use CPU.")

    parser.add_argument("--continue_training_from_last_epoch", action="store_true", help="If set, continue training from the last epoch.")
    
    args = parser.parse_args()
    # Check graph_type and use_literals_message compatibility
    if args.use_literals_message and args.graph_type != LITERALS_GRAPH:
        raise ValueError(f"Literal message passing can only be used with the '{LITERALS_GRAPH}' graph type.")
    # Check cuda availability
    if args.use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your PyTorch installation or use the CPU.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.continue_training_from_last_epoch and args.only_run_test:
        raise ValueError("Cannot continue training from last epoch and only run test at the same time. Please choose one of the two options.")

    # File paths
    ml_dataset_path = args.ml_dataset_path
    dataset_wkdir_path = args.dataset_wkdir_path
    
    
    
    # Reading partitions.pkl
    partitions = pkl.load(open(os.path.join(dataset_wkdir_path, "partitions.pkl"), "rb"))
    partitions = partitions["ml_partitions"]

    # Creating loaders
    train_loader = create_dataloader_from_names(partitions["train"],ml_dataset_path,args.batch_size,graph_type=args.graph_type,shuffle=True)
    if "valid" in partitions  and len(partitions["valid"]) > 0:
        valid_loader = create_dataloader_from_names(partitions["valid"],ml_dataset_path,args.batch_size,graph_type=args.graph_type,shuffle=False)
    else:
        valid_loader = None

    # Output directories
    base_ml_training_path = os.path.join(dataset_wkdir_path, "ml_training")
    os.makedirs(base_ml_training_path, exist_ok=True)
    if args.use_literals_message:
        config_name = f"graph_with_{args.graph_type}_{args.num_layers}_{args.layer_type}_with_literals_message"
    else:
        config_name = f"graph_with_{args.graph_type}_{args.num_layers}_{args.layer_type}"
    config_path = os.path.join(base_ml_training_path, config_name)
    os.makedirs(config_path, exist_ok=True)

    
    training_log_path = os.path.join(config_path, "training_log.csv")
    best_model_path = os.path.join(config_path, "best_model.pth")
    last_model_path = os.path.join(config_path, "last_model.pth")
    last_optimizer_path = os.path.join(config_path, "last_optimizer.pth")

    # Logging setup
    if args.only_run_test:
        logfile = open(training_log_path, 'a')
    else:
        if args.continue_training_from_last_epoch:
            logfile = open(training_log_path, 'a')
        else:
            logfile = open(training_log_path, 'w')
    def add_to_log(epoch,time,partition,metrics):
        logfile.write(",".join(str(val) for val in ([epoch,time,partition]+metrics))+"\n")
        logfile.flush()
    

    # Metrics definition
    # Metrics that aggregate over the 3 classes
    aggregation_metrics_functions = {
        "accuracy_macro":lambda input,target: multiclass_accuracy(input,target,average="macro",num_classes=3),
        "accuracy_micro":lambda input,target: multiclass_accuracy(input,target,average="micro",num_classes=3),
        "f1_score_macro":lambda input,target: multiclass_f1_score(input,target,average="macro",num_classes=3),
        "f1_score_micro":lambda input,target: multiclass_f1_score(input,target,average="micro",num_classes=3),
        "precision_macro":lambda input,target: multiclass_precision(input,target,average="macro",num_classes=3),
        "precision_micro":lambda input,target: multiclass_precision(input,target,average="micro",num_classes=3),
    }
    p = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for p_value in p:
        aggregation_metrics_functions[f"retrieval_precision_{p_value:.2f}"] = create_precision_at_p(p_value)
    # Metrics that are computed for each class
    classes_indexes= {"B0":0,"B1":1,"NB":2}
    confusion_matrix_mapping = {
        f"cm_{row_class}_{col_class}": classes_indexes[row_class] * len(classes_indexes) + classes_indexes[col_class]
        for row_class in classes_indexes
        for col_class in classes_indexes
    }
    individual_metrics_functions = {
        "multiclass_accuracy":(lambda input,target: multiclass_accuracy(input,target,average=None,num_classes=3),classes_indexes),
        "multiclass_precision":(lambda input,target: multiclass_precision(input,target,average=None,num_classes=3),classes_indexes),
        "multiclass_recall":(lambda input,target: multiclass_recall(input,target,average=None,num_classes=3),classes_indexes),
        "multiclass_f1_score":(lambda input,target: multiclass_f1_score(input,target,average=None,num_classes=3),classes_indexes),
        "confusion_matrix":(lambda input,target: multiclass_confusion_matrix(input,target,num_classes=3,normalize="all").flatten(),confusion_matrix_mapping),
    }
    
    metric_aggregator = MetricAggregator(aggregation_metrics_functions,individual_metrics_functions)

    # Model initialization
    model = BackbonePredictor(
        graph_type=args.graph_type,
        layer_type=args.layer_type,
        num_layers=args.num_layers,
        use_literals_message=args.use_literals_message,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print("Model and optimizer initialized.")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    print(model)
    # Training
    if not args.only_run_test:
        if args.continue_training_from_last_epoch:
            print("Continuing training from last epoch.")
            state = torch.load(last_model_path)
            model.load_state_dict(state)
            optimizer_state = torch.load(last_optimizer_path)
            optimizer.load_state_dict(optimizer_state)
            training_log = pd.read_csv(training_log_path)
            starting_epoch = training_log["epoch"].max()+1
            starting_best_loss = training_log[training_log["partition"]=="valid"]["loss"].min()
            print(f"Starting from epoch {starting_epoch} with best loss {starting_best_loss}.")
        else:
            starting_epoch = 0
            starting_best_loss = float("inf")
            print("Starting training from scratch.")
            add_to_log("epoch","time","partition",["loss"]+metric_aggregator.get_metric_names())
        full_train(model,optimizer,args.epochs,best_model_path,last_model_path, last_optimizer_path,add_to_log,metric_aggregator,device,args.batch_accumulation_size,train_loader,valid_loader,starting_epoch=starting_epoch,starting_best_loss=starting_best_loss)

    # Testing
    if "test" not in partitions or len(partitions["test"]) == 0:
        print("No test partition found.")
    else:
        state = torch.load(best_model_path)
        model.load_state_dict(state)
        #check if all files exists
        test_partition_exists = True
        for name in partitions["test"]:
            if not os.path.exists(os.path.join(ml_dataset_path,name+".pkl")):
                test_partition_exists = False
                break
        if not test_partition_exists:
            print(f"Test partition found, but some instances are missing. Skipping test.")
        else:
            print(f"Test partition found, all test instances found. Running test.")
            test_loader = create_dataloader_from_names(partitions["test"],ml_dataset_path,args.batch_size,graph_type=args.graph_type,shuffle=False)
            begin=time.time()
            loss, metrics = one_epoch(model, test_loader, metric_aggregator, device, args.batch_accumulation_size, optimizer=None)
            epoch_time = time.time()-begin
            add_to_log("-1",epoch_time,f"test",[loss]+metrics)
    logfile.close()
if __name__ == '__main__':
    main()