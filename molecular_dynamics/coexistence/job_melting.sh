#!/bin/bash
#SBATCH --job-name=md_melting
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --qos=<YOUR_SLURM_QOS>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# RECONSTRUCTED FILE: doesn't survive in the source project (only submit_melting.sh, which
# calls it, does) -- mirrors ../solid/job_solid.sh's header, calling run-ase.py's actual
# positional-arg signature.
#
# Usage: sbatch job_melting.sh <temperature_K>
# Called by submit_melting.sh once per interface temperature (default: 1200 1300 1400).

TEMP=${1:?"Usage: sbatch job_melting.sh <temperature_K>"}

pwd; hostname; date

module purge
module load miniforge
source activate "<PATH_TO_YOUR_CONDA_ENV>"
export SLURM_CPU_BIND=none
export PYTHON=<PATH_TO_YOUR_CONDA_ENV>/bin/python

mkdir -p RESULTS/MELTING

$PYTHON run-ase.py $TEMP > RESULTS/MELTING/stdout_${TEMP}K.txt 2>&1

date
pwd; hostname; date
