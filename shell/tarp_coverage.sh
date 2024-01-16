#!/bin/bash
#SBATCH --tasks=1
#SBATCH  --array=1-6%6
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=40G        # memory per node
#SBATCH --time=00-02:59   # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=Coverage_data_gen
module load python
source $HOME/diffusion/bin/activate

python $HOME/projects/rrg-lplevass/noedia/bayesian_imaging_radio/tarp-diffusion/scripts/view_tarp.py \
    --method=bootstrapping\
    --uncertainty=3\
    --experiment_name=probes64_postneurips \
    --output_name=""\
    --img_size=64\
    --num_samples=600\
    --samples_dir=/home/noedia/scratch/bayesian_imaging_radio/tarp_samples/euler/ \
    --results_dir=/home/noedia/scratch/bayesian_imaging_radio/tarp_samples/ \
    --num_points=50\