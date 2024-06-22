#!/bin/sh
#
#SBATCH --account=iicd         # Replace ACCOUNT with your group account name
#SBATCH --job-name=LSTM-VAE     # The job name.
#SBATCH -o slurm_out/%j.%N.out
#SBATCH --error=slurm_out/%j.%N.err_out
#SBATCH --gres=gpu:1             # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 2                      # The number of cpu cores to use
#SBATCH --mem-per-cpu=16gb         # The memory the job will use per cpu core
#SBATCH --array=1-20               # Creates an array job with indices from 1 to 3

module load anaconda

# Start the timer
start=$(date +%s)

# Define the arguments for each array index
hidden_size=(64 128 256 512)
latent_size=(4 8 16 32 64)
counter=1

for h in "${hidden_size[@]}"; do
  for l in "${latent_size[@]}"; do
    case $SLURM_ARRAY_TASK_ID in
        $counter)
        ARGS="--hidden_size $h --latent_size $l --model_name LV_EQUATION_LSTM_NORM"
        ;;
    esac
    let counter++
  done
done

#Command to execute Python program
python models/lstmvae/train_vae.py $ARGS
 
# End the timer
end=$(date +%s)

# Calculate the time taken
time_taken=$((end-start))

# print the time taken
echo "End of Training. Time taken : $time_taken seconds."
#End of script