import os
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from catalyst import dl, utils
from fake_fnc import FakeFNC
from bilstm import BiLSTM
from sklearn.model_selection import train_test_split
import numpy as np


criterion = nn.MSELoss()
full_dataset = FakeFNC()
model = BiLSTM(seqlen=full_dataset.seqlen, dim=full_dataset.dim)
optimizer = optim.Adam(model.parameters(), lr=0.02)
full_idx = np.arange(len(full_dataset))
train_idx, test_idx = train_test_split(full_idx, test_size=0.1)
train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

loaders = {
    "train": DataLoader(train_dataset, batch_size=32),
    "valid": DataLoader(test_dataset, batch_size=32),
}

runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=10,
    callbacks=[
        dl.CriterionCallback(metric_key="loss",
                             input_key="logits",
                             target_key="targets"),
    ],
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
)

# model evaluation
metrics = runner.evaluate_loader(
    loader=loaders["valid"],
    callbacks=[dl.CriterionCallback(metric_key="loss",
                                    input_key="logits", target_key="targets")],
)

# model inference
for prediction in runner.predict_loader(loader=loaders["valid"]):
    assert prediction["logits"].detach().cpu().numpy().shape[-1] == 10

# model post-processing
model = runner.model.cpu()
batch = next(iter(loaders["valid"]))[0]
utils.trace_model(model=model, batch=batch)
# utils.quantize_model(model=model)
#utils.prune_model(model=model, pruning_fn="l1_unstructured", amount=0.8)
# utils.onnx_export(model=model, batch=batch, file="./logs/mnist.onnx", verbose=True
