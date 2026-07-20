#!/bin/bash
#SBATCH --job-name=calc_melt
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --qos=<YOUR_SLURM_QOS>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

# Usage: sbatch job_calc_melting.sh [temp1 temp2 ...]
# Example: sbatch job_calc_melting.sh 1200 1300 1400
# If no temperatures are given, calc_melting.py uses its own default list.

pwd; hostname; date

module purge
module load miniforge
source activate "<PATH_TO_YOUR_CONDA_ENV>"
export PYTHON=<PATH_TO_YOUR_CONDA_ENV>/bin/python

if [ $# -gt 0 ]; then
    $PYTHON calc_melting.py --temperatures "$@" > RESULTS/MELTING/calc_melting_stdout.txt 2>&1
else
    $PYTHON calc_melting.py > RESULTS/MELTING/calc_melting_stdout.txt 2>&1
fi

date
pwd; hostname; date
