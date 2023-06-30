#!/usr/bin/env bash

#SBATCH --job-name=pydra_vfp
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=5GB
#SBATCH -o ./slurm/slurm-%A_%a.out

python3 clear_locks.py

echo 'Finished.'