import torch 
import numpy as np
import os
from torch.func import vmap, grad
from tqdm import tqdm
import h5py

from score_models import ScoreModel
import matplotlib.pyplot as plt
import sys
sys.path.append("../models")

from forward_model import model, score_likelihood, link_function
from posterior_sampling import euler_sampler, pc_sampler

device = "cuda" if torch.cuda.is_available() else "cpu"

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args): 
    # Importing and loading the weights of the score of the prior 
    prior = args.prior
    score_model = ScoreModel(checkpoints_directory=prior)
    S = torch.tensor(np.load(args.sampling_function).astype(bool)).to(device)
    img_size = args.model_pixels
    pad = args.pad

    if "probes" in prior:
        print("Running inference with probes 64*64...") 
        B, C = 1/2, 1/2
        model_parameters = (B, C, S, pad)
        

    elif "skirt" in prior: 
        print("Running inference with skirt...")
        B, C = 1, 0
        model_parameters = (B, C, S, pad)
    

    # Sampling parameters
    sampler = args.sampler
    num_pred = args.num_pred
    num_corr = args.num_corr
    snr = args.snr
    test_time = args.test_time
    
    # Number of images sampled per loop
    batch_size = args.batch_size
    num_samples = args.num_samples
    
    path = args.results_dir + f"{sampler}/"

    filename = os.path.join(path, args.experiment_name + f"_{THIS_WORKER}" + ".h5")

    print("Creating a ground-truth...")
    ground_truth = score_model.sample([1, 1, args.model_pixels, args.model_pixels], steps=num_pred)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("model", [args.num_samples, 1, args.model_pixels, args.model_pixels], dtype=np.float32)

        
        observation = model(t = torch.zeros(1).to(device), 
                            x = ground_truth.flatten(),
                            score_model = score_model,
                            model_parameters = model_parameters)
        sigma_y = args.sigma_likelihood
        observation += torch.randn_like(observation) * sigma_y

        hf.create_dataset("reconstruction", [args.num_samples, observation.shape[0]], dtype=np.float32)
        hf["observation"] = observation.cpu().numpy().astype(np.float32).squeeze()
        hf["ground_truth"] = link_function(ground_truth).cpu().numpy().astype(np.float32).squeeze()
        
        
        for i in range(int(num_samples//batch_size)):
            if sampler.lower() == "euler":
                print("Starting posterior sampling with Euler-Maruyama...")    
                samples = euler_sampler(
                    y = observation,
                    sigma_y = sigma_y,
                    forward_model = model, 
                    score_model = score_model,
                    score_likelihood = score_likelihood, 
                    model_parameters = model_parameters,
                    num_samples = batch_size,
                    num_steps = num_pred,  
                    img_size = img_size,
                    test_time = test_time
                )
                

            elif sampler.lower() == "pc":
                # pc_params = [(1000, 10, 1e-2), (1000, 100, 1e-2), (1000, 1000, 1e-3)]
                # #pc_params = [(1000, 1000, 1e-3)]
                # idx = int(THIS_WORKER//100)
                # pred, corr, snr = pc_params[idx]

                print(f"Sampling pc pred = {num_pred}, corr = {num_corr}, snr = {snr}")
                samples = pc_sampler(
                    y = observation,
                    sigma_y = sigma_y,
                    num_samples = batch_size,
                    num_pred_steps = num_pred,
                    num_corr_steps = num_corr,
                    snr = snr,
                    score_function = score_posterior,
                    img_size = img_size
                )
                
                
            else : 
                raise ValueError("The sampler specified is not implemented or does not exist. Choose between 'euler' and 'pc'")
            
            hf["model"][i*batch_size: (i+1)*batch_size] = samples.cpu().numpy().astype(np.float32)

            # Let's hope it doesn't take too much time compared to the posterior sampling:
            y_hat = torch.empty(size = (batch_size, 1, img_size, img_size)).to(device)
            for j in range(batch_size):
                y_hat = model(t = torch.zeros(1).to(device), 
                              x = samples[j],
                              score_model = score_model, 
                              model_parameters = model_parameters)
            hf["reconstruction"][i*batch_size: (i+1)*batch_size] = y_hat.cpu().numpy().astype(np.float32)



if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Likelihood parameters
    parser.add_argument("--sigma_likelihood",   required = True,                    type = float,   help = "The square root of the multiplier of the isotropic gaussian matrix")
    
    # Experiments spec
    parser.add_argument("--results_dir",        required = True,                                    help = "Directory where to save the TARP files")
    parser.add_argument("--experiment_name",    required = True,                                    help = "Prefix for the name of the file")
    

    parser.add_argument("--model_pixels",       required = True,                    type = int)
    
    # Sampling parameters
    parser.add_argument("--sampler",            required = False,   default = "pc", type = str,     help = "Sampling procedure used ('pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 20,   type = int,     help = "Number of samples from the posterior to create")
    parser.add_argument("--batch_size",         required = False,   default = 20,   type = int)
    parser.add_argument("--num_pred",           required = False,   default = 1000, type = int,     help ="Number of iterations in the loop to compute the reverse sde")
    parser.add_argument("--num_corr",           required = False,   default = 20,   type = int,     help ="Number of corrector steps for the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2, type = float)
    parser.add_argument("--pad",                required = False,   default = 0,    type = int)
    parser.add_argument("--sampling_function",  required = True)
    parser.add_argument("--prior",              required = True)
    parser.add_argument("--test_time",          required = False,   default = False,type = bool)
    
    args = parser.parse_args()
    main(args) 




