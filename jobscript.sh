#!/bin/bash -l
#
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=job123
#SBATCH -o /home/hpc/iwb3/iwb3111h/output/slurm-%j.out
#SBATCH -e /home/hpc/iwb3/iwb3111h/output/slurm-%j.err
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
module load python/3.10-anaconda
source activate train
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python create_fixed_list.py