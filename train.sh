#!/bin/sh
#
#SBATCH --account=iicd         # Replace ACCOUNT with your group account name
#SBATCH --job-name=LSTM-VAE     # The job name.
#SBATCH -o slurm_out/%j.%N.out
#SBATCH --error=slurm_out/%j.%N.err_out
#SBATCH --gres=gpu:1             # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 4                      # The number of cpu cores to use
#SBATCH --mem-per-cpu=8gb         # The memory the job will use per cpu core
 
module load anaconda
 
#Command to execute Python program
python models/lstmvae/train_vae.py
 
#End of script