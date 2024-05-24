import os
import glob
import tqdm
import pandas as pd
import numpy as np

root_dir = "/data/users3/bbaker/projects/Dynamic_BrainAge/data/tcwindows_v2"
csv_dir = os.path.join(root_dir, 'csvs')

#all_csvs = list(glob.glob(os.path.join(csv_dir, "*")))

all_dfs = []
missing = []
N = 22569
for f in tqdm.tqdm(range(N)):
    expected_fn = os.path.join(csv_dir, "sample_%d_ws-240.pth.csv" % f)
    if not os.path.exists(expected_fn):
        missing.append(f)
        continue
    left, right = expected_fn.split("_ws")
    left, right = right.split(".pth")
    left = left[1:]
    wsize = int(left)
    df = pd.read_csv(expected_fn)
    df['wsize'] = [30,60,120,240]
    all_dfs.append(df)

full_df = pd.concat(all_dfs).reset_index(drop=True)

full_df.to_csv("/data/users3/bbaker/projects/Dynamic_BrainAge/data/tcwindows_converted_files_v2.csv")
with open("/data/users3/bbaker/projects/Dynamic_BrainAge/data/tcwindows_missing.txt", "w") as file:
    for line in missing:
        file.write("%s\n" % str(line)) 