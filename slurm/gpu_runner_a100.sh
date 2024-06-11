#!/bin/bash
#SBATCH -p qTRDGPUL,qTRDGPUM,qTRDGPUH
#SBATCH --mem=300gb
#SBATCH -c 16
#SBATCH --gres=gpu:A100:1
#SBATCH -t 48:00:00
#SBATCH -A trends53c17
#SBATCH --oversubscribe
hostname
#ls /data/users3
#ls /data/collaboration
echo ARGS "${@:1}"
eval "$(conda shell.bash hook)"
conda activate brainage_ni24
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker43/anaconda3/lib python -u main.py "${@:1}"

