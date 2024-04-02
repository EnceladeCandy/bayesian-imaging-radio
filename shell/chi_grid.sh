#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=40G               # memory per node
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=chi_samples_skirt
#SBATCH --output=%x-%j.out

# To run this part, the script must be executed directly (e.g. source script.sh or ./script.sh)
# then the loop will run and submit multiple jobs with different hyperparameters
if [ "$SLURM_JOB_USER" == "" ]; then
    for M in 1 10; do
        for SNR in 0.1 0.01; do  # Add or modify Tdm values as needed
            sbatch ./chi_grid.sh $M $SNR
        done
    done
    exit 0
fi


source $HOME/gingakei/bin/activate
M=$1
SNR=$2

RESULTS_DIR_SKIRT=/home/noedia/scratch/bayesian_imaging_radio/tarp_experiment/gridsearch/post_sampling_cl/vpskirt64
RESULTS_DIR_PROBES=/home/noedia/scratch/bayesian_imaging_radio/tarp_experiment/gridsearch/post_sampling_cl/veprobes64
SCRIPTS_DIR=/home/noedia/projects/rrg-lplevass/noedia/bayesian_imaging_radio/bayesian-imaging-radio/scripts

# Posterior sampling
python $SCRIPTS_DIR/generate_chi_squared.py \
    --grid=True\
    --predictor=4000\
    --corrector=$M\
    --snr=$SNR\
    --experiment_dir=$RESULTS_DIR_SKIRT/pc\
    --results_dir=$RESULTS_DIR_SKIRT\