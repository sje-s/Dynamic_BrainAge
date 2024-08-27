import os
import torch
import argparse
import json
import tqdm
import inspect
import numpy as np
import copy
import pandas as pd

from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import KFold

from dynamic_brainage.dataloaders.get_dataset import get_dataset
from dynamic_brainage.dataloaders.CSVDataset import CSVDataset, get_subset
from dynamic_brainage.models.get_model import get_model
from dynamic_brainage.defaults.default_args import DEFAULTS, HELP
from dynamic_brainage.defaults.testing import DEFAULTS, HELP

from scipy.io import savemat

import warnings
warnings.filterwarnings("ignore")

# Begin Argument Parsing
parser = argparse.ArgumentParser("Dynamic Models for BrainAge")
for key, val in DEFAULTS.items():
    parser.add_argument("--%s" % key, default=val,
                        type=type(val), help=HELP[key])
args = parser.parse_args()
old_logdir = args.logdir
i = 1
while os.path.exists(args.logdir):
    args.logdir = old_logdir + "_" + str(i)
    i += 1
os.makedirs(args.logdir, exist_ok=True)
json.dump(args.__dict__, open(os.path.join(
    args.logdir, "parameters.json"), "w"))
# Set seed before ANYTHING
torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

device = "cuda" if torch.cuda.is_available() else "cpu"
# Resolve Pytorch SubModules
args.optimizer = getattr(torch.optim, args.optimizer)
args.criterion = getattr(torch.nn, args.criterion)
if args.scheduler is not None:
    if hasattr(torch.optim.lr_scheduler, args.scheduler):
        args.scheduler = getattr(torch.optim.lr_scheduler, args.scheduler)
    else:
        args.scheduler = None
# Resolve dataset and model
full_train_dataset = None
if args.train_dataset is not None:
    full_train_dataset = get_dataset(args.train_dataset,
                                     *json.loads(args.train_dataset_args),
                                     **json.loads(args.train_dataset_kwargs))
full_inference_dataset = None
if args.test_dataset is not None:
    if args.test_dataset.lower() != "valid":
        full_inference_dataset = get_dataset(args.test_dataset,
                                             *json.loads(args.test_dataset_args),
                                             **json.loads(args.test_dataset_kwargs))
        
        
model = get_model(args.model, 
                  *json.loads(args.model_args),
                  **json.loads(args.model_kwargs)).to(device)
# Resolve Metrics
args.train_metrics = json.loads(args.train_metrics)
args.test_metrics = json.loads(args.test_metrics)
# END ARGUMENT PARSING

# initialize criterion and optimizer
criterion = args.criterion()
os.makedirs(os.path.join(args.logdir, "logs"), exist_ok=True)
os.makedirs(os.path.join(args.logdir, "checkpoints"), exist_ok=True)
if full_train_dataset is not None:
    sig = inspect.signature(args.optimizer)
    optim_kwargs = {k: v for k, v in json.loads(
        args.optim_kwargs).items() if k in sig.parameters.keys()}
    optimizer = args.optimizer(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **optim_kwargs)
    scheduler = None
    if args.scheduler is not None:
        sig = inspect.signature(args.scheduler)
        sched_args = json.loads(args.scheduler_args)
        sched_kwargs = {k: v for k, v in json.loads(
                        args.scheduler_kwargs).items() 
                        if k in sig.parameters.keys()}
        scheduler = args.scheduler(optimizer,
                                   *sched_args, 
                                   **sched_kwargs)
    full_idx = np.arange(full_train_dataset.N_subjects)
    kfold = KFold(n_splits=args.num_folds,
                  shuffle=True, random_state=args.seed)
    for k, (train_idx, valid_idx) in enumerate(kfold.split(full_idx)):
        if k == args.k:
            break
    if isinstance(full_train_dataset, CSVDataset):
        train_dataset = get_subset(full_train_dataset, train_idx)
        valid_dataset = get_subset(full_train_dataset, valid_idx)
    else:
        train_dataset = Subset(full_train_dataset, train_idx)
        valid_dataset = Subset(full_train_dataset, valid_idx)
    if args.test_dataset.lower() == "valid":
        full_inference_dataset = copy.deepcopy(valid_dataset)
    #full_inference_dataset = dataset_with_indices(full_inference_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=14, prefetch_factor=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=14, prefetch_factor=2, shuffle=True)
    # model training
    rows_batches = []
    rows_epochs = []
    step = 0
    training_rows_accumulated = []
    validation_rows_accumulated = []
    training_rows = []
    validation_rows = []
    best_loss = torch.inf
    
    if not args.inference_only:
        for epoch in range(args.epochs):
            running_loss = []
            running_corr = []
            model.train()
            pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
            print("\n***Training Epoch %d/%d***\n" % (epoch, args.epochs))
            for batch_i, batch in pbar:
                optimizer.zero_grad()
                for j, bi in enumerate(batch):
                    batch[j] = bi.to(device)
                # for now assume tuple
                x, y, _ = batch
                y = y.float()
                yhat = model(x)
                loss = criterion(yhat.view_as(y), y)
                stacked = torch.stack([yhat, y.view_as(yhat)], 0).squeeze()
                corr = torch.corrcoef(stacked).flatten()[1]
                if torch.isnan(corr).item():
                    corr = torch.Tensor([0.])
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                training_rows.append(dict(step=step, 
                                        epoch=epoch, 
                                        batch_index=batch_i, 
                                        loss=loss.item(), 
                                        corr=corr.item()))
                running_loss.append(loss.item())
                running_corr.append(corr.item())
                step += 1
                pbar.set_description("%d/%d     Loss:%.6f     :%.4f+-%.3f     Corr:%.6f:%.4f±%.3f     " % (batch_i+1, 
                                                                                            len(train_loader), 
                                                                                            running_loss[-1], 
                                                                                            np.mean(running_loss), 
                                                                                            np.std(running_loss), 
                                                                                            running_corr[-1],
                                                                                            np.mean(running_corr),
                                                                                            np.std(running_corr)))
            training_rows_accumulated.append(dict(
                epoch=epoch,
                loss_mean=np.mean(running_loss),
                loss_std=np.std(running_loss),
                corr_mean=np.mean(running_corr),
                corr_std=np.std(running_corr),
            ))
            
            pd.DataFrame(training_rows).to_csv(os.path.join(args.logdir, "logs", "train_full.csv"), index=False)
            pd.DataFrame(training_rows_accumulated).to_csv(os.path.join(args.logdir, "logs", "train.csv"), index=False)
            running_loss = []
            running_corr = []
            model.eval()
            print("\n***Validation Epoch %d/%d***\n" % (epoch, args.epochs))
            with torch.no_grad():
                pbar = tqdm.tqdm(enumerate(valid_loader))
                for batch_i, batch in pbar:
                    for j, bi in enumerate(batch):
                        batch[j] = bi.to(device)
                    # for now assume tuple
                    x, y, _ = batch
                    y = y.float()
                    yhat = model(x)
                    loss = criterion(yhat.view_as(y), y)
                    stacked = torch.stack([yhat, y.view_as(yhat)], 0).squeeze()
                    corr = torch.corrcoef(stacked).flatten()[1]
                    if torch.isnan(corr).item():
                        corr = torch.Tensor([0.])


                    validation_rows.append(dict(step=step, 
                                            epoch=epoch, 
                                            batch_index=batch_i, 
                                            loss=loss.item(), 
                                            corr=corr.item()))
                    running_loss.append(loss.item())
                    running_corr.append(corr.item())
                    step += 1        
                    pbar.set_description("%d/%d     Loss:%.6f     :%.4f+-%.3f     Corr:%.6f:%.4f±%.3f     " % (batch_i+1, 
                                                                                                len(valid_loader), 
                                                                                                running_loss[-1], 
                                                                                                np.mean(running_loss), 
                                                                                                np.std(running_loss), 
                                                                                                running_corr[-1],
                                                                                                np.mean(running_corr),
                                                                                                np.std(running_corr)))
            validation_rows_accumulated.append(dict(epoch=epoch,
                loss_mean=np.mean(running_loss),
                loss_std=np.std(running_loss),
                corr_mean=np.mean(running_corr),
                corr_std=np.std(running_corr),
            ))
            pd.DataFrame(validation_rows).to_csv(os.path.join(args.logdir, "logs", "valid_full.csv"), index=False)
            pd.DataFrame(validation_rows_accumulated).to_csv(os.path.join(args.logdir, "logs", "valid.csv"), index=False)
            os.makedirs(os.path.join(args.logdir, "checkpoints"), exist_ok=True)
            if np.mean(running_loss) < best_loss:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.mean(running_loss)
                    }, 
                    os.path.join(args.logdir, "checkpoints", "best.pth")
                )
                best_loss = np.mean(running_loss)
                print("Saved at epoch %d" % epoch)

if full_inference_dataset is not None:
    test_loader = DataLoader(full_inference_dataset,
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=12, 
                             prefetch_factor=2)
    new_model = get_model(args.model, 
                  *json.loads(args.model_args),
                  **json.loads(args.model_kwargs)).to(device)
    test_rows_accumulated = []
    test_rows = []
    if "<EVAL>" in args.inference_model:
        args.inference_model = eval(args.inference_model.replace("<EVAL>",""))
    checkpoint = torch.load(args.inference_model)
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_model.eval()
    # add callbacks
    callbacks = []
    all_indices = []
    all_predictions = []
    all_deltas = []
    running_loss = []
    running_corr = []
    print("\n***Inference***\n")
    step = 0
    pbar = tqdm.tqdm(enumerate(test_loader))
    for batch_i, batch in pbar:
        for j, bi in enumerate(batch):
            batch[j] = bi.to(device)
        # for now assume tuple
        x, y, idx = batch
        all_indices.append(idx.detach().cpu().numpy())
        y = y.float()
        yhat = new_model(x)
        all_deltas.append((y-yhat.view_as(y)).detach().cpu().numpy())
        all_predictions.append(yhat.detach().cpu().numpy())
        loss = criterion(yhat.view_as(y), y)
        stacked = torch.stack([yhat, y.view_as(yhat)], 0).squeeze()
#         corr = torch.corrcoef(stacked).flatten()[1]
#         if torch.isnan(corr).item():
#             corr = torch.Tensor([0.])
        test_rows.append(dict(step=step, 
                                epoch=0, 
                                batch_index=batch_i, 
                                loss=loss.item()))
#         test_rows.append(dict(step=step, 
#                                 epoch=0, 
#                                 batch_index=batch_i, 
#                                 loss=loss.item(), 
#                                 corr=corr.item()))
        running_loss.append(loss.item())
#         running_corr.append(corr.item())
        step += 1        
#         pbar.set_description("%d/%d     Loss:%.6f     :%.4f+-%.3f     Corr:%.6f:%.4f±%.3f     " % (batch_i+1, 
        pbar.set_description("%d/%d     Loss:%.6f     :%.4f+-%.3f     Corr:N/A     " % (batch_i+1, 
                                                                                    len(test_loader), 
                                                                                    running_loss[-1], 
                                                                                    np.mean(running_loss), 
                                                                                    np.std(running_loss)))
#                                                                                     running_corr[-1],
#                                                                                     np.mean(running_corr),
#                                                                                     np.std(running_corr)))
        test_rows_accumulated.append(dict(epoch=0,
            loss_mean=np.mean(running_loss),
            loss_std=np.std(running_loss),
#             corr_mean=np.mean(running_corr),
#             corr_std=np.std(running_corr),
        ))
        pd.DataFrame(test_rows).to_csv(os.path.join(args.logdir, "logs", "test_full.csv"), index=False)
        pd.DataFrame(test_rows_accumulated).to_csv(os.path.join(args.logdir, "logs", "test.csv"), index=False)
    all_predictions = np.concatenate(all_predictions, 0)
#     all_deltas = np.concatenate(all_deltas, 0)
    all_indices = np.concatenate(all_indices, 0)
    sidx = np.argsort(all_indices)
    all_predictions = all_predictions[sidx]
#     all_deltas = all_deltas[sidx]
#     predict_df = pd.DataFrame([dict(subject=sub,
#                  session=ses,
#                  label=l,
#                  prediction=p[0],
#                  delta=d,
#                  filepath=f) for (sub, ses, l, p, d, f) in zip(
#                      full_inference_dataset.subjects,
#                      full_inference_dataset.sessions,
#                      full_inference_dataset.labels,
#                      all_predictions,
#                      all_deltas,
#                      full_inference_dataset.file_paths,
#                  )])
#     predict_df.to_csv(os.path.join(args.logdir,"logs", "predictions.csv"), index=False)
    savemat(os.path.join(args.logdir,"logs", "predictions.mat"), {"preds": all_predictions})