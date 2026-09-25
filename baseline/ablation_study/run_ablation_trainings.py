import subprocess
import argparse
import os

parser = argparse.ArgumentParser(prog='run_ablation_trainings',
                    description='Trains models for an ablation study')
parser.add_argument('--ablation_configs_path', type=str, required = True, help="Path to the directory containing the ablation config files")

args = parser.parse_args()

ablation_configs_path = args.ablation_configs_path

config_files = [f for f in os.listdir(ablation_configs_path)]
config_files.sort()

for f in config_files:
    training_conf = os.path.join(ablation_configs_path, f)
    cmd = ["python3", "train_ablation.py", "--task", training_conf]
                
    process = subprocess.run(cmd, stderr=subprocess.STDOUT, text=True, check=True)
