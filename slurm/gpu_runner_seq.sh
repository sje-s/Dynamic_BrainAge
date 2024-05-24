#!/bin/bash
#SBATCH -p qTRDGPUH,qTRDGPUM,qTRDGPUL,qTRDGPU
#SBATCH -c 12
#SBATCH --mem=32gb
#SBATCH --gres=gpu:A100:1
#SBATCH -t 24:00:00
#SBATCH -A trends53c17
#SBATCH --oversubscribe
hostname
echo ARGS "${@:1}"
eval "$(conda shell.bash hook)"
conda activate catalyst3.9
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker43/anaconda3/lib python main_seq.py "${@:1}"

