#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=4G               # memory per node
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=<>
#SBATCH --output=%x-%j.out

# To run this part, the script must be executed directly (e.g. source script.sh or ./script.sh)
# then the loop will run and submit multiple jobs with different hyperparameters


if [ "$SLURM_JOB_USER" == "" ]; then
    for SNR in 0.1 0.01; do
        for M in 1 10; do
            sbatch $0 $SNR $M 
        done
    done
    exit 0
fi

M=$2
SNR=$1
python $HOME/projects/rrg-lplevass/noedia/bayesian_imaging_radio/tarp-diffusion/scripts/test_grid.py \
    --predictor=4000\
    --corrector=$M\
    --snr=$SNR
