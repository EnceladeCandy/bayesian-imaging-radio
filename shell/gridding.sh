#!/bin/bash
#SBATCH --tasks=1
#SBATCH  --array=0-20%20
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32G        # memory per node
#SBATCH --time=00-02:59   # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=Gridding_DSHARP
module load python
source $HOME/diffusion/bin/activate

TARGETS_DIR=/home/noedia/scratch/bayesian_imaging_radio/dsharp_npz
RESULTS_DIR=/home/noedia/scratch/bayesian_imaging_radio/dsharp_gridded
SCRIPT_DIR=/home/noedia/projects/def-lplevass/noedia/bayesian_imaging_radio/bayesian-imaging-radio/src/preprocessing

python $SCRIPT_DIR/gridding.py \
    --npz_dir=$TARGETS_DIR \
    --output_dir=$RESULTS_DIR \
    --npix=4096\
    --window_function="sinc"\
    --img_size=256
