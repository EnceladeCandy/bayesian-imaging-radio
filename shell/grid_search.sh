#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --array=1-10%2
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G               # memory per node
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=<>
#SBATCH --output=%x-%j.out

# To run this part, the script must be executed directly (e.g. source script.sh or ./script.sh)
# then the loop will run and submit multiple jobs with different hyperparameters
if [ "$SLURM_JOB_USER" == "" ]; then
    for SNR in 0.1 0.01; do
        for M in 1 5 10; do
            sbatch $0 $SNR $M
        done
    done
    exit 0
fi

M=$2
SNR=$1
python $HOME/projects/rrg-lplevass/noedia/bayesian_imaging_radio/tarp-diffusion/scripts/inference_sim.py \
    --sigma_y=1e-2\
    --results_dir=/home/noedia/scratch/bayesian_imaging_radio/tarp_samples/ \
    --experiment_name=skirt64_pc\
    --model_pixels=64\
    --sampler=pc\
    --num_samples=500\
    --batch_size=250\
    --num_pred=4000\
    --num_corr=20\
    --snr=1e-1\
    --pad=96\
    --sampling_function=$HOME/projects/rrg-lplevass/data/sampling_function3.npy \
    --prior=$HOME/projects/rrg-lplevass/data/score_models/ncsnpp_vp_skirt_y_64_230813225149