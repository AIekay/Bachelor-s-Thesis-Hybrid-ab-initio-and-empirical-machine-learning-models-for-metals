#!/bin/bash
#SBATCH --job-name=md_solid
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --qos=<YOUR_SLURM_QOS>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Usage: sbatch job_solid.sh <temperature_K>
# Example: sbatch job_solid.sh 800

TEMP=${1:-800}

pwd; hostname; date

module purge
module load miniforge
source activate "<PATH_TO_YOUR_CONDA_ENV>"
export SLURM_CPU_BIND=none
export PYTHON=<PATH_TO_YOUR_CONDA_ENV>/bin/python

mkdir -p RESULTS/SOLID/logs RESULTS/SOLID/xyz

$PYTHON md_solid.py --temperature $TEMP > RESULTS/SOLID/logs/stdout_${TEMP}K.txt 2>&1

date
pwd; hostname; date
