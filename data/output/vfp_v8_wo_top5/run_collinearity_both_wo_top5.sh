#!/usr/bin/env bash

#SBATCH --job-name=pydra_vfp
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH -c 25
#SBATCH --mem=10GB
#SBATCH -o ./slurm/slurm-%A.out

pydraml -s specs/vfp_spec_4models_both_wo_top5.json

echo 'Finished.'