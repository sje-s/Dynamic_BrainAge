# BrainAge with Dynamic Models

Single-layer LSTM which predicts BrainAge.
This is just a starting example.

## Dependencies

Make sure you have needed dependencies, either by doing pip install
```
pip install -r requirements.txt
```
or better, creating a conda environment:
```
conda env create -f environment.yml
```

## Running


**NOTE:** The examples are currently out of date, so using main.py is recommended until I find time to fix the examples.

The full code can be run using main.py and the arguments below:

```
$ python main.py --help
usage: RNNs for BrainAge [-h] [--devices DEVICES] [--criterion CRITERION] [--train-dataset TRAIN_DATASET] [--train-dataset-args TRAIN_DATASET_ARGS]
                         [--train-dataset-kwargs TRAIN_DATASET_KWARGS] [--test-dataset TEST_DATASET] [--test-dataset-args TEST_DATASET_ARGS]
                         [--test-dataset-kwargs TEST_DATASET_KWARGS] [--model MODEL] [--model-args MODEL_ARGS] [--model-kwargs MODEL_KWARGS] [--inference-model INFERENCE_MODEL]
                         [--optimizer OPTIMIZER] [--optim-kwargs OPTIM_KWARGS] [--batch-size BATCH_SIZE] [--lr LR] [--weight-decay WEIGHT_DECAY] [--num-folds NUM_FOLDS]
                         [--epochs EPOCHS] [--train-metrics TRAIN_METRICS] [--test-metrics TEST_METRICS] [--seed SEED] [--logdir LOGDIR] [--k K]

optional arguments:
  -h, --help            show this help message and exit
  --devices DEVICES     Environment variable defining available GPU devices
  --criterion CRITERION
                        Loss function for training/validation - must be a module name in torch.nn
  --train-dataset TRAIN_DATASET
                        Dataset to use for training <fakefnc/ukb/ukbhcp1200>
  --train-dataset-args TRAIN_DATASET_ARGS
                        Arguments to pass to training dataset (JSON parsed list)
  --train-dataset-kwargs TRAIN_DATASET_KWARGS
                        Keyword Arguments to pass to training dataset (JSON parsed dict)
  --test-dataset TEST_DATASET
                        Dataset to use for inference <fakefnc/ukb/ukbhcp1200/valid> - 'valid' uses the validation data
  --test-dataset-args TEST_DATASET_ARGS
                        Arguments to pass to inference dataset (ignored if validation)
  --test-dataset-kwargs TEST_DATASET_KWARGS
                        Keyword Arguments to pass to inference dataset (ignored if validation)
  --model MODEL         Model to use <bilstm/bilstm-classifier/bilstm-hierarchical>
  --model-args MODEL_ARGS
                        Arguments to pass to model (JSON parsed list)
  --model-kwargs MODEL_KWARGS
                        Keyword arguments to pass to model (JSON parsed dict)
  --inference-model INFERENCE_MODEL
                        Model name to load for inference (looks in logdir/checkpoints)
  --optimizer OPTIMIZER
                        Optimizer to use for training (must be found in torch.optim)
  --optim-kwargs OPTIM_KWARGS
                        Keyword arguments for Optimizer
  --batch-size BATCH_SIZE
                        Batch size for both training and inference
  --lr LR               Learning rate for training (float)
  --weight-decay WEIGHT_DECAY
                        Weight decay for training (float)
  --num-folds NUM_FOLDS
                        Number of Cross-Validation Folds
  --epochs EPOCHS       The number of epochs to train
  --train-metrics TRAIN_METRICS
                        The metrics to compute for training (JSON parsed list) <loss/correlation/auc>
  --test-metrics TEST_METRICS
                        The metrics to compute for testing (JSON parsed list) <loss/correlation/auc>
  --seed SEED           The seed to initialize ALL RANDOM STATES
  --logdir LOGDIR       The logging directory, where results are saved (created upon runtime)
  --k K                 The current fold to evaluate
  ```

  ## Inference

  To use a pretrained model, simply set the `--train-dataset` argument to NONE and specify a test data set you wish to run on. You will also need to point the `--inference-model` argument to the pretrained model file. You will also need to make sure the `--model` and `--model-args` and `--model-kwargs` agree with those used for creating the pretrained model. 
