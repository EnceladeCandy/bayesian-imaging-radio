#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --array=1-500%100
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=40G               # memory per node
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --gres=gpu:1
#SBATCH --job-name=gridsearch_pc_veprobes
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


source $HOME/gingakei/bin/activate
M=$1
SNR=$2

# Posterior sampling
python $HOME/projects/rrg-lplevass/noedia/bayesian_imaging_radio/tarp-diffusion/scripts/view_tarp.py \
    --method=bootstrapping\
    --uncertainty=3\
    --experiment_name=vpskirt64 \
    --grid=true\
    --corrector=$M\
    --snr=$SNR\
    --output_name=""\
    --img_size=64\
    --num_samples=600\
    --samples_dir=/home/noedia/scratch/bayesian_imaging_radio/tarp_samples/euler/ \
    --results_dir=/home/noedia/scratch/bayesian_imaging_radio/tarp_samples/ \
    --num_points=50\
