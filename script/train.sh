#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=dill-sage
#SBATCH --time=4000
#SBATCH --output=/home/tangyimi/controllable-seo/slurm_output/%j.out
#SBATCH --job-name=nosys

python -m experiment.main 
