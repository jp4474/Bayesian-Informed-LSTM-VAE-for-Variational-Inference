#!/bin/sh
#
#SBATCH --account=iicd         # Replace ACCOUNT with your group account name
#SBATCH --job-name=LSTM-VAE     # The job name.
#SBATCH -o slurm_out/%j.%N.out
#SBATCH --error=slurm_out/%j.%N.err_out
#SBATCH -c 4                      # The number of cpu cores to use
#SBATCH --mem-per-cpu=8gb         # The memory the job will use per cpu core

module load anaconda

# Start the timer
start=$(date +%s)

#Command to execute Python program
python vi.py

# End the timer
end=$(date +%s)

# Calculate the time taken
time_taken=$((end-start))

# print the time taken
echo "End of Mosaic run. Time taken : $time_taken seconds." 
# End of file