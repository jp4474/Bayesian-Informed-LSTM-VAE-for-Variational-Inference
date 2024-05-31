#!/bin/sh
#
#
#SBATCH --account=iicd         # Replace ACCOUNT with your group account name
#SBATCH --job-name=LSTM-VAE     # The job name.
#SBATCH --gres=gpu:1             # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 1                      # The number of cpu cores to use
#SBATCH -t 0-0:30                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=8gb         # The memory the job will use per cpu core
 
module load anaconda
 
#Command to execute Python program
python main.py
 
#End of script