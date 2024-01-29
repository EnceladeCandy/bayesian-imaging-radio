"""
Objective: 
Simulate an observation and generate posterior samples associated to a known ground-truth (a simulation). 
The ground-truth is sampled from a score-based model and a forward model simulates an observation 
from a radio interferometer with isotropic gaussian noise (see https://arxiv.org/pdf/2311.18012.pdf 
for more information on the forward model used).

This code creates an h5 file containing the ground truth, the associated posterior samples, the observation
and the reconstructions obtained by forward modeling each posterior sample. 
"""

import os
import json
import torch 
from torch.func import vmap, grad
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from utils import create_dir
from score_models import ScoreModel

import sys
sys.path.append("../src/inference")

from forward_model import model, score_likelihood, link_function
from posterior_sampling import euler_sampler, old_pc_sampler, pc_sampler

device = "cuda" if torch.cuda.is_available() else "cpu"

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args): 
    print(f"Generating posterior samples for simulation {THIS_WORKER}/{N_WORKERS}")

    # Importing and loading the weights of the score of the prior 
    prior = args.prior
    score_model = ScoreModel(checkpoints_directory=prior)
    sampling_function = torch.tensor(np.load(args.sampling_function))[:256**2]
    S = sampling_function.reshape(256, 256).to(device)
    img_size = args.model_pixels
    pad = args.pad

    # Setting the parameters for the score-based prior
    if "probes" in prior:
        print(f"Running inference with probes {img_size}*{img_size}...") 
        B, C = 1/2, 1/2
        model_parameters = (S, B, C, pad)
        dataset = "probes"
        
    elif "skirt" in prior: 
        print(f"Running inference with skirt {img_size}*{img_size}...")
        B, C = 1, 0
        model_parameters = (S, B, C, pad)
        dataset = "skirt"


    # Sampling parameters
    sampler = args.sampler
    predictor = args.predictor
    corrector = args.corrector
    snr = args.snr
    debug_mode = args.debug_mode

    # Number of images sampled per loop
    batch_size = args.batch_size
    num_samples = args.num_samples
    
    # Euler sampling of the ground-truth.
    ground_truth = score_model.sample([1, 1, args.model_pixels, args.model_pixels], steps=predictor)    
    total_samples = np.empty(shape = [args.num_samples, 1, args.model_pixels, args.model_pixels], dtype = np.float32)
    observation = model(t = torch.zeros(1).to(device), 
                        x = ground_truth[0],
                        score_model = score_model,
                        model_parameters = model_parameters)
    sigma_y = args.sigma_y
    observation += sigma_y * torch.randn_like(observation)

    reconstruction = np.empty(shape = [args.num_samples, observation.shape[-1]])
    
    # Generating posterior samples
    for i in range(int(num_samples//batch_size)):
        if sampler.lower() == "euler":
            print("Using Euler-Maruyama sampler...")    
            samples = euler_sampler(
                y = observation,
                sigma_y = sigma_y,
                forward_model = model, 
                score_model = score_model,
                score_likelihood = score_likelihood, 
                model_parameters = model_parameters,
                num_samples = batch_size,
                num_steps = predictor,  
                img_size = (img_size, img_size),
                keep_chain = False,
                debug_mode = debug_mode
            )

        
        elif sampler.lower() == "pc": 
            print(f"Using pc sampler: {predictor} predictor steps | {corrector} corrector steps | snr = {snr}")
            sampling_params = (predictor, corrector, snr)
            samples = pc_sampler(
                y = observation,
                sigma_y = sigma_y, 
                forward_model = model, 
                score_model = score_model, 
                score_likelihood = score_likelihood, 
                model_parameters = model_parameters,
                num_samples = batch_size,
                pc_params = sampling_params,
                img_size = (img_size, img_size), 
                keep_chain = False, 
                debug_mode = debug_mode 
            )
            
        
        # Forward modeling posterior samples 
        y_hat = model(t = torch.zeros(1).to(device), 
                        x = samples,
                        score_model = score_model, 
                        model_parameters = model_parameters) # (1, N_vis)
        
        total_samples[i * batch_size: (i+1) * batch_size] = samples.cpu().numpy().astype(np.float32)
        reconstruction[i * batch_size: (i+1) * batch_size] = y_hat.squeeze().cpu().numpy().astype(np.float32)
    
    # Creating experiment's directory
    print("Creating folder for the experiment in the results directory...")
    path_experiment = os.path.join(args.results_dir, args.experiment_name)
    create_dir(path_experiment)
    
    # Creating sampler directory
    print("Creating folder for the sampler used in the experiment's directory...")
    path_sampler = os.path.join(path_experiment, sampler)
    create_dir(path_sampler)


    # Creating directory according to the pc parameter being used
    print("Creating folder for the parameters used in the sampler's directory")
    if sampler == "pc": 
        params_foldername = f"{predictor}pred_{corrector}corr_{snr}snr"
    elif sampler == "euler": 
        params_foldername = f"{predictor}steps"
    path = os.path.join(path_sampler, params_foldername)
    create_dir(path)

    
    file_dir = os.path.join(path, f"samples_sim_{THIS_WORKER}.npz")
    
    np.savez(
            file_dir,  
            ground_truth = link_function(ground_truth, B, C).cpu().numpy().astype(np.float32).squeeze(), 
            samples = total_samples, 
            observation = observation.cpu().numpy().astype(np.float32).squeeze(), 
            reconstruction = reconstruction
        )
    
    if args.sanity_plot: 
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize = (8, 4))
        im = axs[0].imshow(link_function(ground_truth[0], B, C).squeeze().cpu(), cmap = "magma")
        plt.colorbar(im, ax = axs[0])
        axs[0].set_title("Ground-truth")
        im = axs[1].imshow(link_function(samples[0], B, C).squeeze().cpu(), cmap = "magma")
        plt.colorbar(im, ax = axs[1])
        axs[1].set_title("Posterior sample")

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
                "experiment_name": args.experiment_name, 
                "dataset": dataset,
                "sde": sde,
                "sampler": sampler,
                "num_samples": num_samples,
                "num_sims": N_WORKERS,
                "model_pixels": img_size, 
                "sampling_params": list(sampling_params),
                "sigma_y": sigma_y  
            }
            filename = "params.json"
            with open(args.results_dir + filename, "w") as json_file: 
                json.dump(data, json_file, indent = 2)


if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Likelihood parameters
    parser.add_argument("--sigma_y",            required = True,                        type = float,   help = "The square root of the multiplier of the isotropic gaussian matrix")
    parser.add_argument("--sampling_function",  required = True,                        type = str,     help = "Directory of the sampling function (generated via the gridding code")
    
    # Experiments spec
    parser.add_argument("--results_dir",        required = True,                                        help = "Directory where to save the TARP files")
    parser.add_argument("--experiment_name",    required = True,                                        help = "Prefix for the name of the file")
    parser.add_argument("--model_pixels",       required = True,                        type = int,     help = "Image size (only supporting images with equal width and heights for now). The total number of pixels in the image should be model_pixels * model_pixels")
    
    # Sampling parameters
    parser.add_argument("--sampler",            required = False,   default = "euler",  type = str,     help = "Sampler used ('old_pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 20,       type = int,     help = "Total number of posterior samples to generate.")
    parser.add_argument("--batch_size",         required = False,   default = 20,       type = int,     help = "Number of posterior samples to generate per iteration (the code begins a loop if num_samples > batch_size).")
    parser.add_argument("--predictor",          required = False,   default = 1000,     type = int,     help = "Number of steps if sampler is 'euler'. Number of predictor steps if the sampler is 'pc'")
    parser.add_argument("--corrector",          required = False,   default = 20,       type = int,     help = "Number of corrector steps for the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2,     type = float,   help = "Parameter pc sampling")
    parser.add_argument("--pad",                required = False,   default = 0,        type = int,     help = "Padding (must respect the sampling function size)")
    parser.add_argument("--prior",              required = True)
    parser.add_argument("--debug_mode",         required = False,   default = False,    type = bool,    help = "True to skip loops and debug")
    parser.add_argument("--sanity_plot",        required = False,   default = False,    type = bool,    help = "True to create a plot with posterior samples and the ground truth (if testing the script put both debug_mode and sanity_plot to True)")

    parser.add_argument("--save_params",        required = False,   default = True,     type = bool,   help = "True in order to save the experiment's parameters in a json file.")
    
    args = parser.parse_args()
    main(args) 




