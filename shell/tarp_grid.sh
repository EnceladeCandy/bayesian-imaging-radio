#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=40G               # memory per node
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=coverage_data_gen
#SBATCH --output=%x-%j.out

# To run this part, the script must be executed directly (e.g. source script.sh or ./script.sh)
# then the loop will run and submit multiple jobs with different hyperparameters
if [ "$SLURM_JOB_USER" == "" ]; then
    for M in 1 10; do
        for SNR in 0.1 0.01; do  # Add or modify Tdm values as needed
            sbatch ./tarp_grid.sh $M $SNR
        done
    done
    exit 0
fi


source $HOME/diffusion/bin/activate
M=$1
SNR=$2

RESULTS_DIR=/home/noedia/scratch/bayesian_imaging_radio/tarp_experiment/gridsearch/post_sampling_cl/vpskirt64

# Posterior sampling
python generate_tarp_data.py \
    --mode=posterior\
    --grid=True\
    --predictor=4000\
    --corrector=$M\
    --snr=$SNR\
    --experiment_dir=$RESULTS_DIR/pc\
    --method=bootstrapping\
    --num_points=50\
    --results_dir=$RESULTS_DIR\
    --sanity_plot=True
    

