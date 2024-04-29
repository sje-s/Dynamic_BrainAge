#!/bin/bash
#SBATCH -p qTRDGPU,qTRDGPUL,qTRDGPUM,qTRDGPUH
#SBATCH --mem=20gb
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -A trends53c17
#SBATCH --oversubscribe
hostname
ls /data/users3
ls /data/collaboration
echo ARGS "${@:1}"
eval "$(conda shell.bash hook)"
conda activate catalyst3.9
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker43/anaconda3/lib python main.py "${@:1}"

