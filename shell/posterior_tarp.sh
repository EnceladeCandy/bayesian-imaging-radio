#!/bin/bash
#SBATCH --tasks=1
#SBATCH --array=0-500%100
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G         # memory per node
#SBATCH --time=00-11:59   # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Posterior_probes_postneurips
module load python
source $HOME/diffusion/bin/activate

# SKIRT = ncsnpp_vp_skirt_y_64_230813225149
# PROBES = ncsnpp_probes_g_64_230604024652
python $HOME/projects/rrg-lplevass/noedia/bayesian_imaging_radio/tarp-diffusion/scripts/inference_sim.py \
    --sigma_y=1e-2\
    --sampling_function=$HOME/projects/rrg-lplevass/data/sampling_function3.npy \
    --prior=$HOME/projects/rrg-lplevass/data/score_models/ncsnpp_probes_g_64_230604024652 \
    --pad=96\
    --model_pixels=64\
    --sampler=euler\
    --num_pred=4000\
    --num_samples=600\
    --batch_size=300\
    --results_dir=/home/noedia/scratch/bayesian_imaging_radio/tarp_samples/ \
    --experiment_name=probes64_postneurips