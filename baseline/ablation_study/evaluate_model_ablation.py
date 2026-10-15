import sys
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from scipy import stats
import pandas as pd
from pandas import DataFrame as df
import argparse
import yaml
import os

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from rnn import RNNModel
from dataset_rnn_ablation import TrajectoryDataset, collate_fn


parser = argparse.ArgumentParser(prog='evaluate_model',
                    description='Evaluates the specified model on a given dataset.')
parser.add_argument('--model', type=str, required = True, help="Model trained for rating prediction.")
parser.add_argument('--context', type=str, required = True,  help="Context file for the generation of the context variables.")
parser.add_argument('--dataset', type=str, required = True, help="Dataset file used for testing the model (TXT)")
parser.add_argument('--dataroot', type=str, required = True, help="Data root directory.")
parser.add_argument('--trainconfig', type=str, required = True, help="YAML file used for training the model.")
parser.add_argument('--saveresults', action='store_true', help='Save the results')

args = parser.parse_args()

config_data = yaml.load(open(args.trainconfig, "r"), Loader=yaml.Loader)

FEATURES_TO_REMOVE = config_data["FEATURES_TO_REMOVE"]


def concordance_corrcoef(y_true, y_pred):
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2)


checkpoint_path = args.model
print(checkpoint_path)
model_name = checkpoint_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

input_size = checkpoint['input_size']
hidden_size = checkpoint['hidden_size']
num_layers = checkpoint['num_layers']

if 'rnn_type' in checkpoint.keys():
    RNN_TYPE = checkpoint['rnn_type']
else:
    RNN_TYPE = 'GRU'
if 'frame_threshold' in checkpoint.keys():
    FRAME_THRESHOLD = checkpoint['frame_threshold']
else:
    FRAME_THRESHOLD = 0.1

if 'linear_layers' in checkpoint.keys():
    LINEAR_LAYERS = checkpoint['linear_layers']
else:
    LINEAR_LAYERS = []

if 'activation' in checkpoint.keys():
    ACTIVATION = checkpoint['activation']
else:
    ACTIVATION = 'linear'

if 'context_features' in checkpoint.keys():
    CONTEXT_FEATURES = checkpoint['context_features']
else:
    CONTEXT_FEATURES = 0


model = RNNModel(input_size, hidden_size, num_layers, rnn_type = RNN_TYPE, linear_layers = LINEAR_LAYERS, activation=ACTIVATION, context_vars=CONTEXT_FEATURES).to(device)
    
model = model.to(device)


model.load_state_dict(checkpoint['model_state_dict'], strict=True)

datafile = args.dataset
contextQ_file = args.context

dataset = TrajectoryDataset(datafile, contextQ_file, path = args.dataroot, limit=-1, frame_threshold=FRAME_THRESHOLD, data_augmentation=False, reload = False, features_to_remove = FEATURES_TO_REMOVE)
loader = DataLoader(dataset,batch_size=32, shuffle=False, collate_fn=collate_fn)

mse_function = torch.nn.MSELoss().to(device)
mae_function = torch.nn.L1Loss().to(device)

model.eval()
loss_mse = 0
loss_mae = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for val_data in loader:
        traj_batch, label_batch, seq_length = val_data
        label_batch = label_batch.to(device)
        traj_batch = traj_batch.to(device)
        seq_length = seq_length.to(device)

        predictions = model(traj_batch, seq_length).squeeze()
        loss_mse += mse_function(predictions, label_batch).item() * len(label_batch)
        loss_mae += mae_function(predictions, label_batch).item() * len(label_batch)
        all_predictions += predictions.tolist()
        all_labels += label_batch.tolist()
    
loss_mse /= len(dataset)
loss_mae /= len(dataset)

pearson_correlation = stats.pearsonr(all_predictions, all_labels)
spearman_correlation = stats.spearmanr(all_predictions, all_labels)
ccc = concordance_corrcoef(all_predictions, all_labels)

if args.saveresults and os.path.exists('ablation_study.csv'):
    ablation_results_DF = pd.read_csv('ablation_study.csv', index_col = 0)
    ablation_results = ablation_results_DF.to_dict('index')
else:
    ablation_results = dict()

ablation_results[args.model] = dict()
ablation_results[args.model]['MSE'] = loss_mse
ablation_results[args.model]['MAE'] = loss_mae
ablation_results[args.model]['Pearson Corr'] = pearson_correlation.statistic
ablation_results[args.model]['pvalue Pearson'] = pearson_correlation.pvalue
ablation_results[args.model]['Spearman Corr'] = spearman_correlation.statistic
ablation_results[args.model]['pvalue Spearman'] = spearman_correlation.pvalue
ablation_results[args.model]['ccc'] = ccc


ablation_results_DF = df.from_dict(ablation_results, orient='index')
print(ablation_results_DF)
if args.saveresults:
    ablation_results_DF.to_csv("ablation_study.csv")