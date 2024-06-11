#!/bin/bash
#SBATCH -p qTRD
#SBATCH --mem=2gb
#SBATCH -c 1
#SBATCH -t 1:00:00
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH -J fft
#SBATCH -e /data/users3/bbaker/projects/Dynamic_BrainAge/slurm/clogs/convert_%A_%a.err
#SBATCH -o /data/users3/bbaker/projects/Dynamic_BrainAge/slurm/clogs/convert_%A_%a.out
hostname
#ls /data/users3
#ls /data/collaboration 
eval "$(conda shell.bash hook)"
cd /data/users3/bbaker/projects/Dynamic_BrainAge
export PYTHONPATH="/data/users3/bbaker/projects/Dynamic_BrainAge"
conda activate brainage_ni24
#export DATA_CHUNK_OFFSET=$1
OFFSET=$1
k=$((SLURM_ARRAY_TASK_ID + OFFSET))
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker43/anaconda3/lib python scripts/conversion/convert_all_spectrograms.py --k $k
