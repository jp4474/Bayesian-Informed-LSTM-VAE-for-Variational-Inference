#!/bin/sh
#
#
#SBATCH --account=iicd         # Replace ACCOUNT with your group account name
#SBATCH --job-name=LSTM-VAE     # The job name.
#SBATCH -c 1                      # The number of cpu cores to use
#SBATCH -t 0-0:30                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=8gb         # The memory the job will use per cpu core
 
module load anaconda
 
#Command to execute Python program
python generate_training_data.py
 
#End of script