#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=dill-sage
#SBATCH --mem=32G
#SBATCH --time=4000
#SBATCH --output=/home/tangyimi/controllable-seo/slurm_output/ragroll/%j.out
#SBATCH --job-name=transfer

python -m experiment.transfer --base=mistral-7b
