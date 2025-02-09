#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=glamor-ruby
#SBATCH --time=4000
#SBATCH --output=/home/tangyimi/controllable-seo/slurm_output/%j.out
#SBATCH --job-name=mistral

python -m experiment.main --mode=suffix
# python -m experiment.main --mode=paraphrase
