import os
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import argparse
import json
import glob
import numpy as np

parser = argparse.ArgumentParser("LSTM-BrainAge Plots")
parser.add_argument("--experiment", default="test")
parser.add_argument("--result-dir", default="figures")
parser.add_argument("--font-scale", default=2)
#parser.add_argument(
#    "--plots", default='["lineplot:step X loss;{}", "lineplot:step X correlation/mean;{}"]')
parser.add_argument(
    "--plots", default='["lineplot;{}"]')
args = parser.parse_args()
plots = json.loads(args.plots)
outdir = os.path.join(args.result_dir, args.experiment)
os.makedirs(outdir, exist_ok=True)
logdir = os.path.join("logs", args.experiment)
predictions = []
prediction_path = os.path.join(logdir,"predictions","predictions.txt")
if os.path.exists(prediction_path):
    with open(prediction_path, "r") as file:
        for i, line in enumerate(file):
            ls = line.split(" ")
            for t, bas in enumerate(ls):
                row = dict(subject="Subject %d" % i, brainage=float(bas), time=t)
                predictions.append(row)
predict_df = pd.DataFrame(predictions)

sb.set(font_scale=1.5)
fig, ax = plt.subplots(1,1,figsize=(10,10))
sb.lineplot(x="time", y="brainage", hue="subject", data=predict_df)
plt.ylabel("BrainAge")
plt.xlabel("Time")
plt.savefig(os.path.join("figures", "dynamic_brainage_test.png"), bbox_inches="tight")