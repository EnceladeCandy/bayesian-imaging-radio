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
    for M in 1; do
        for SNR in 0.1 0.01; do  # Add or modify Tdm values as needed
            sbatch ./grid_search.sh $M $SNR
        done
    done
    exit 0
fi

SCRIPTS=$HOME/projects/rrg-lplevass/noedia/bayesian_imaging_radio/bayesian-imaging-radio/scripts
RESULTS_DIR=$HOME/scratch/bayesian_imaging_radio/tarp_experiment/gridsearch/post_sampling_cl
SHARED_DATA=$HOME/projects/rrg-lplevass/data
SKIRT64=$SHARED_DATA/score_models/ncsnpp_vp_skirt_y_64_230813225149
PROBES64=$SHARED_DATA/ncsnpp_probes_g_64_230604024652

source $HOME/gingakei/bin/activate
NUM_SAMPLES=500
B=500
N=4000 # predictor steps
M=$1
SNR=$2

# Posterior sampling
python $SCRIPTS/inference_sim.py \
    --sigma_y=1e-2\
    --results_dir=$RESULTS_DIR \
    --experiment_name=veprobes64 \
    --model_pixels=64\
    --sampler=pc\
    --num_samples=$NUM_SAMPLES\
    --batch_size=$B\
    --predictor=$N\
    --corrector=$M\
    --snr=$SNR\
    --pad=96\
    --sampling_function=$SHARED_DATA/sampling_function3.npy \
    --prior=$PROBES64 \
    --save_params=True
