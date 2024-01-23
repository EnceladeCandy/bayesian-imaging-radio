"""
Author: Noé Dia and Alexandre Adam 

Objective: 
Simulate an observation and generate posterior samples associated to a known ground-truth. 
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
from tqdm import tqdm
import h5py

from score_models import ScoreModel
import matplotlib.pyplot as plt
import sys
sys.path.append("../models")

from forward_model import model, score_likelihood, link_function
from posterior_sampling import euler_sampler, old_pc_sampler, pc_sampler

device = "cuda" if torch.cuda.is_available() else "cpu"

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))


def create_dir(dir): 
    if not os.path.exists(dir): 
        os.makedirs(dir)

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
    num_pred = args.num_pred
    num_corr = args.num_corr
    snr = args.snr
    debug_mode = args.debug_mode

    # Number of images sampled per loop
    batch_size = args.batch_size
    num_samples = args.num_samples
    
    
    # ground_truth = score_model.sample([1, 1, args.model_pixels, args.model_pixels], steps=num_pred)
    if sampler == "euler": 
        sampling_params = num_pred
        ground_truth = score_model.sample([1, 1, args.model_pixels, args.model_pixels], steps=sampling_params)
    
    elif sampler == "old_pc": 
        sampling_params = (num_pred, num_corr, snr)
        ground_truth = old_pc_sampler(
            y = None, 
            sigma_y = None, 
            forward_model = None, 
            score_model = score_model,
            score_likelihood = None, 
            model_parameters = None, 
            num_samples = 1, 
            pc_params = sampling_params, 
            img_size = (img_size, img_size), 
            keep_chain = False, 
            debug_mode = debug_mode
        )
    
    elif sampler == "pc": 
        sampling_params = (num_pred, num_corr, snr)
        ground_truth = pc_sampler(
            y = None, 
            sigma_y = None, 
            forward_model = None, 
            score_model = score_model,
            score_likelihood = None, 
            model_parameters = None, 
            num_samples = 1, 
            pc_params = sampling_params, 
            img_size = (img_size, img_size), 
            keep_chain = False, 
            debug_mode = debug_mode
        )
    else : 
         raise ValueError("The sampler specified is not implemented or does not exist. Choose between 'euler' and 'pc'")
    
    
    total_samples = np.empty(shape = [args.num_samples, 1, args.model_pixels, args.model_pixels], dtype = np.float32)
    #hf.create_dataset("model", [args.num_samples, 1, args.model_pixels, args.model_pixels], dtype=np.float32)

    observation = model(t = torch.zeros(1).to(device), 
                        x = ground_truth[0],
                        score_model = score_model,
                        model_parameters = model_parameters)
    sigma_y = args.sigma_y
    observation += sigma_y * torch.randn_like(observation)

    reconstruction = np.empty(shape = [args.num_samples, observation.shape[-1]])
    # hf.create_dataset("reconstruction", [args.num_samples, observation.shape[0]], dtype=np.float32)
    # hf["observation"] = observation.cpu().numpy().astype(np.float32).squeeze()
    # hf["ground_truth"] = link_function(ground_truth, B, C).cpu().numpy().astype(np.float32).squeeze()
    
    
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
                num_steps = num_pred,  
                img_size = (img_size, img_size),
                keep_chain = False,
                debug_mode = debug_mode
            )

        elif sampler.lower() == "old_pc":
            # pc_params = [(1000, 10, 1e-2), (1000, 100, 1e-2), (1000, 1000, 1e-3)]
            # #pc_params = [(1000, 1000, 1e-3)]
            # idx = int(THIS_WORKER//100)
            # pred, corr, snr = pc_params[idx]

            print(f"Sampling pc pred = {num_pred}, corr = {num_corr}, snr = {snr}")
            
            samples = old_pc_sampler(
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
        
        elif sampler.lower() == "pc": 
            print(f"Using pc sampler: {num_pred} predictor steps| {num_corr} corrector steps| snr = {snr}")
            sampling_params = (num_pred, num_corr, snr)
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
            
        
        
        #hf["model"][i*batch_size: (i+1)*batch_size] = link_function(samples.cpu().numpy().astype(np.float32), B, C)

        # Forward modeling posterior samples 
        y_hat = model(t = torch.zeros(1).to(device), 
                        x = samples,
                        score_model = score_model, 
                        model_parameters = model_parameters)
        print(reconstruction.shape)
        total_samples[i * batch_size: (i+1) * batch_size] = samples.cpu().numpy().astype(np.float32)
        reconstruction[i * batch_size: (i+1) * batch_size] = y_hat.squeeze().cpu().numpy().astype(np.float32)
        # hf["reconstruction"][i*batch_size: (i+1)*batch_size] = y_hat.cpu().numpy().astype(np.float32)
    
    path = args.results_dir + f"{sampler}/"
    create_dir(path)
    filename = os.path.join(path, args.experiment_name + f"_{THIS_WORKER}_{num_pred}pred_{num_corr}corr_{snr}snr" + ".npz")
    np.savez(
            filename,  
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
        
        plt.savefig("/home/noedia/scratch/bayesian_imaging_radio/tarp_samples/sanity_posterior.jpeg", bbox_inches = "tight")
    
    # Saving a json file with the script parameters 
    if THIS_WORKER==1: 
        print("Saving the experiment's parameters...")
        if "ve" in str(score_model.sde): 
            sde = "VE"
        else: 
            sde = "VP"
        data = { 
            "num_samples": num_samples,
            "num_sims": N_WORKERS,
            "model_pixels": img_size, 
            "dataset": dataset,
            "sde": sde, 
            "experiment_name": args.experiment_name, 
            "sampling_params": list(sampling_params),
            "sigma_y": sigma_y
        }
        filename = path+ f"/{args.experiment_name}.json"
        with open(filename, "w") as json_file: 
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
    parser.add_argument("--model_pixels",       required = True,                        type = int)
    
    # Sampling parameters
    parser.add_argument("--sampler",            required = False,   default = "euler",  type = str,     help = "Sampler used ('old_pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 20,       type = int,     help = "Number of samples from the posterior to create")
    parser.add_argument("--batch_size",         required = False,   default = 20,       type = int)
    parser.add_argument("--num_pred",           required = False,   default = 1000,     type = int,     help ="Number of iterations in the loop to compute the reverse sde")
    parser.add_argument("--num_corr",           required = False,   default = 20,       type = int,     help ="Number of corrector steps for the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2,     type = float)
    parser.add_argument("--pad",                required = False,   default = 0,        type = int)
    parser.add_argument("--prior",              required = True)
    parser.add_argument("--debug_mode",         required = False,   default = False,    type = bool)
    parser.add_argument("--sanity_plot",        required = False,   default = False,    type = bool)
    
    args = parser.parse_args()
    main(args) 




