#!/usr/bin/env bash

#SBATCH --job-name=pydra_vfp
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem=10GB
#SBATCH -o ./slurm/slurm-%A.out

pydraml -s specs/vfp_spec_4models_speech_if_4-39_1-out-of-5.json
pydraml -s specs/vfp_spec_4models_speech_if_4-39_2-out-of-5.json
pydraml -s specs/vfp_spec_4models_speech_if_4-39_3-out-of-5.json
pydraml -s specs/vfp_spec_4models_speech_if_4-39_4-out-of-5.json
pydraml -s specs/vfp_spec_4models_speech_if_4-39_5-out-of-5.json


echo 'Finished.'