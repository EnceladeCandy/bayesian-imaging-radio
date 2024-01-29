#!/bin/bash

# Declare SNR and M as integers
declare -i SNR M

# SLURM parameters for the array jobs (assuming 3 SNR values and 3 M values)
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00-03:00
#SBATCH --account=def-lplevass
#SBATCH --job-name=my_grid_search_%A_%a_SNR${SNR}_M${M}
#SBATCH --output=%x-%j.out

# Assign SNR and M values based on array job index
# (Assuming SNR values are 0.1, 0.01, 0.001, represented as integers 10, 1, 0)
let "SNR = (SLURM_ARRAY_TASK_ID - 1) / 3"
let "SNR_float = SNR / 100"  # Calculate floating-point SNR for Python script
let "M = (SLURM_ARRAY_TASK_ID - 1) % 3 + 1"

python $HOME/projects/rrg-lplevass/noedia/bayesian_imaging_radio/tarp-diffusion/scripts/test_grid.py \
    --predictor=4000 \
    --corrector=$M \
    --snr=$SNR_float  # Pass the calculated floating-point SNR
