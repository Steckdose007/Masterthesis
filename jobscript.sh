#!/bin/bash -l
#SBATCH --job-name=example
#SBATCH --time=8:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH -o /home/hpc/…/username/output/slurm-%j.out
#SBATCH -e /home/hpc/…/username/output/slurm-%j.err
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
module load python/3.9-anaconda
source activate my_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python hyperparametertuning.py