#!/bin/bash

# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --array=1-25%25
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=40G               # memory per node
#SBATCH --time=00-03:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --gres=gpu:1
#SBATCH --job-name=prior_sampling_veprobes
#SBATCH --output=%x-%j.out

# To run this part, the script must be executed directly (e.g. source script.sh or ./script.sh)
# then the loop will run and submit multiple jobs with different hyperparameters
SCRIPTS=$HOME/projects/rrg-lplevass/noedia/bayesian_imaging_radio/bayesian-imaging-radio/scripts
RESULTS_DIR=$HOME/scratch/bayesian_imaging_radio/tarp_experiment/prior_sampling
SHARED_DATA=$HOME/projects/rrg-lplevass/data
SKIRT64=$SHARED_DATA/score_models/ncsnpp_vp_skirt_y_64_230813225149

# ON NARVAL THE DIRECTORY'S NOT THE SAME
PROBES64=$SHARED_DATA/ncsnpp_probes_g_64_230604024652

source $HOME/gingakei/bin/activate
NUM_SAMPLES=1000
B=500 # batch size
N=4000 # predictor steps


# Posterior sampling
python $SCRIPTS/generate_prior_samples.py \
    --results_dir=$RESULTS_DIR \
    --experiment_name=veprobes64 \
    --model_pixels=64\
    --num_samples=$NUM_SAMPLES\
    --batch_size=$B\
    --num_steps=$N\
    --prior=$PROBES64 \
    --sanity_plot=True\