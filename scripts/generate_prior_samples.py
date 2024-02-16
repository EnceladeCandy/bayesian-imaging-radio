import os
import json
import torch 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import create_dir
from score_models import ScoreModel

import sys
sys.path.append("../src/inference")
from forward_model import link_function


device = "cuda" if torch.cuda.is_available() else "cpu"

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args): 
    if N_WORKERS>1:
        print(f"Generating prior samples for simulation {THIS_WORKER}/{N_WORKERS}")

    # Importing and loading the weights of the score of the prior 
    prior = args.prior
    score_model = ScoreModel(checkpoints_directory=prior)
    img_size = args.model_pixels

    # Setting the parameters for the score-based prior
    if "probes" in prior:
        print(f"Creating prior samples with probes {img_size}*{img_size}...") 
        B, C = 1/2, 1/2
        dataset = "probes"
        
    elif "skirt" in prior: 
        print(f"Creating prior samples with skirt {img_size}*{img_size}...")
        B, C = 1, 0
        dataset = "skirt"


    # Sampling parameters
    predictor = args.num_steps
    debug_mode = args.debug_mode
    batch_size = args.batch_size # Number of images sampled per loop
    num_samples = args.num_samples # Total number of samples to generate
    
    total_samples = np.empty(shape = [args.num_samples, 1, args.model_pixels, args.model_pixels], dtype = np.float32)
    
    # Generating prior samples
    for i in range(int(num_samples//batch_size)):
        samples = score_model.sample([batch_size, 1, args.model_pixels, args.model_pixels], steps=predictor)  
        total_samples[i * batch_size: (i+1) * batch_size] = samples.cpu().numpy().astype(np.float32)
    
    # Creating experiment's directory
    print("Creating folder for the experiment in the results directory...")
    path_experiment = os.path.join(args.results_dir, args.experiment_name)
    create_dir(path_experiment)
    

    # Creating directory according to the pc parameter being used
    print("Creating folder for the parameters used in the sampler's directory")
    params_foldername = f"{predictor}steps"
    path = os.path.join(path_experiment, params_foldername)
    create_dir(path)

    file_dir = os.path.join(path, f"prior_samples_{THIS_WORKER}.npy")
    np.save(file_dir,  link_function(total_samples, B, C))
    
    # Creating a plot to make sure the sampling went well
    if args.sanity_plot: 
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, figsize = (4, 4))
        im = axs.imshow(link_function(samples[0], B, C).squeeze().cpu(), cmap = "magma")
        plt.colorbar(im)
        axs.set_title("Ground-truth")
        image_dir = os.path.join(path_experiment, "sanity.jpeg")
        plt.savefig(image_dir, bbox_inches = "tight")
    
    # Saving a json file with the script parameters 
    if args.save_params: 
        if THIS_WORKER==1: 
            print("Saving the experiment's parameters...")
            if "ve" in str(score_model.sde): 
                sde = "VE"
            elif "vp" in str(score_model.sde): 
                sde = "VP"
            
            else:
                sde = "Unknown"

            data = { 
                "num_samples": num_samples,
                "model_pixels": img_size, 
                "num_steps": predictor
            }
            filename = "params.json"
            params_dir = os.path.join(path, filename)
            with open(params_dir, "w") as json_file: 
                json.dump(data, json_file, indent = 2)


if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    # Experiments spec
    parser.add_argument("--results_dir",        required = True,                                        help = "Directory where to save the TARP files")
    parser.add_argument("--experiment_name",    required = True,                                        help = "Prefix for the name of the file")
    parser.add_argument("--model_pixels",       required = True,                        type = int,     help = "Image size (only supporting images with equal width and heights for now). The total number of pixels in the image should be model_pixels * model_pixels")
    
    # Sampling parameters
    parser.add_argument("--num_samples",        required = False,   default = 20,       type = int,     help = "Total number of posterior samples to generate.")
    parser.add_argument("--batch_size",         required = False,   default = 20,       type = int,     help = "Number of posterior samples to generate per iteration (the code begins a loop if num_samples > batch_size).")
    parser.add_argument("--num_steps",          required = False,   default = 1000,     type = int,     help = "Number of steps if sampler is 'euler'. Number of predictor steps if the sampler is 'pc'")
    parser.add_argument("--prior",              required = True)
    parser.add_argument("--debug_mode",         required = False,   default = False,    type = bool,    help = "True to skip loops and debug")
    parser.add_argument("--sanity_plot",        required = False,   default = False,    type = bool,    help = "True to create a plot with posterior samples and the ground truth (if testing the script put both debug_mode and sanity_plot to True)")
    parser.add_argument("--save_params",        required = False,   default =False)
    args = parser.parse_args()
    main(args) 




