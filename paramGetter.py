import json
import sys

with open(sys.argv[2] + "parameters.json") as f:
    temp = json.load(f)

tp = sys.argv[1]
if (tp == "model"):
    print(temp["model"])
elif (tp == "args"):
    print(temp["model_args"])
elif (tp == "kwargs"):
    print(temp["model_kwargs"])