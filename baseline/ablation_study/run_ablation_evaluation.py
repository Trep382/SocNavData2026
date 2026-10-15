import subprocess
import argparse
import os


parser = argparse.ArgumentParser(prog='run_evaluate_model_ablation',
                    description='Evaluates ablation models')
parser.add_argument('--models_path', type=str, required = True, help="Path to the directory containing the models' files")
parser.add_argument('--context', type=str, required = True,  help="Context file for the generation of the context variables.")
parser.add_argument('--dataset', type=str, required = True, help="Dataset file used for testing the model (TXT)")
parser.add_argument('--dataroot', type=str, required = True, help="Data root directory.")
parser.add_argument('--trainconfig_path', type=str, default='', help="Path to the directory containing the YAML files used for training.")
parser.add_argument('--saveresults', action='store_true', help='Save the results')

args = parser.parse_args()

models_path = args.models_path
config_path = args.trainconfig_path
context = args.context
dataset = args.dataset
dataroot = args.dataroot

model_files = [f for f in os.listdir(models_path) if f.endswith('.pytorch')]
model_files.sort()
config_files = [f for f in os.listdir(config_path) if f.endswith('.yaml')]
config_files.sort()

if len(model_files)!=len(config_files):
    print(f'Something went wrong. The number of model files {len(model_files)} should be the same than the number of configuration files {len(config_files)}')
    exit()

for model, config in zip(model_files,config_files):
    model_file = os.path.join(models_path, model)
    config_file = os.path.join(config_path, config)
    print(f'Evaluating model {model_file}')
    cmd = ["python3", "evaluate_model_ablation.py", "--model", model_file, "--context", context, "--dataset", dataset,
           "--dataroot", dataroot, "--trainconfig", config_file]
    if args.saveresults:
        cmd += '--saveresults'
               
    process = subprocess.run(cmd, stderr=subprocess.STDOUT, text=True, check=True)
