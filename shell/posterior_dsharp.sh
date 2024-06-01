#!/bin/bash
#SBATCH --tasks=1
#SBATCH --array=0-20%20
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G         # memory per node
#SBATCH --time=00-03:00   # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Posterior_samples_DSHARP
module load python
source $HOME/diffusion/bin/activate

# SKIRT = ncsnpp_vp_skirt_y_64_230813225149
# PROBES = ncsnpp_probes_g_64_230604024652
SCRIPT_DIR=/home/noedia/projects/def-lplevass/noedia/bayesian_imaging_radio/bayesian-imaging-radio/scripts
SKIRT=/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_vp_skirt_z_256_230813225243
PROBES=/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_ve_probes_z_256_230926020329

INPUT_DIR=/home/noedia/scratch/bayesian_imaging_radio/dsharp_gridded
RESULTS_DIR=/home/noedia/scratch/bayesian_imaging_radio/dsharp_reconstructions

python $SCRIPT_DIR/inference_dsharp.py \
    --npz_dir=$INPUT_DIR \
    --output_dir=$RESULTS_DIR \
    --img_size=256\
    --npix=4096\
    --sampler=pc\
    --num_samples=5\
    --batch_size=5\
    --predictor=4000\
    --corrector=1\
    --snr=0.1\
    --score_dir=$SKIRT \
    --sanity_plot=True

    
    
    
    