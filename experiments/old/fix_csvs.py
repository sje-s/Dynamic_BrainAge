import os
import glob
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default="RE0001_first_model")
args = parser.parse_args()

RES_DIR = "/data/users3/bbaker/projects/LSTM_BrainAge/logs"

exp_dir = os.path.join(RES_DIR, args.experiment)
if not os.path.exists(exp_dir):
    raise(FileExistsError("Path %s does not exist" % exp_dir))

run_dirs = list(glob.glob(os.path.join(exp_dir, "run*")))

for run_dir in run_dirs:
    log_dir = os.path.join(run_dir, "logs")
    if not os.path.exists(log_dir):
        continue
    train_csv = os.path.join(log_dir,"train.csv")
    valid_csv = os.path.join(log_dir,"valid.csv")
    if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
        continue
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    running_rows = []
    with open(train_csv) as file:
        new_file = False
        