#!/bin/bash
#SBATCH -p qTRDGPU
#SBATCH --mem=20gb
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH --exclude arctrdagn044
hostname
grid_file=$1
echo grid_file $grid_file
eval "$(conda shell.bash hook)"
conda activate catalyst3.9
ARGS=`sed -n "$(( $SLURM_ARRAY_TASK_ID )) p" $grid_file`
echo ARGS $ARGS
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker43/anaconda3/lib python main.py $ARGS