#!/bin/bash
#SBATCH --job-name=train_hybrid
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --qos=<YOUR_SLURM_QOS>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

pwd; hostname; date

module purge
module load miniforge
source activate "<PATH_TO_YOUR_CONDA_ENV>"
export SLURM_CPU_BIND=none
export PYTHON=<PATH_TO_YOUR_CONDA_ENV>/bin/python

$PYTHON train_hybrid.py --steps=5000000 > log.txt
$PYTHON plot_train_evo.py
$PYTHON test.py

date
