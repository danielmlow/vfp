#!/usr/bin/env bash

#SBATCH --job-name=pydra_vfp
#SBATCH --array=1-9
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH -c 25
#SBATCH --mem=10GB
##SBATCH -o ./slurm/slurm-%A_%a.out

pydraml -s specs/vfp_spec_4models_vowel_if_${SLURM_ARRAY_TASK_ID}.json

echo 'Finished.'