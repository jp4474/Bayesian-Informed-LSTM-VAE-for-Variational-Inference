#!/bin/sh
#
#
#SBATCH --account=iicd         # Replace ACCOUNT with your group account name
#SBATCH --job-name=compute_statistics     # The job name.
#SBATCH -c 1                      # The number of cpu cores to use
#SBATCH -t 0-0:30                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=8gb         # The memory the job will use per cpu core
 
module load anaconda
 
#Command to execute Python program
python compute_statistics.py
 
#End of script