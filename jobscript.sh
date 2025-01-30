#!/bin/bash -l
#
#SBATCH --time=48:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --job-name=mel
#SBATCH -o /home/hpc/iwb3/iwb3111h/output/slurm-%j.out
#SBATCH -e /home/hpc/iwb3/iwb3111h/output/slurm-%j.err
#SBATCH --export=NONE
#SBATCH --qos=a100_aibe
unset SLURM_EXPORT_ENV
module load python/3.10-anaconda
source activate train
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
python hyperparametertuning.py