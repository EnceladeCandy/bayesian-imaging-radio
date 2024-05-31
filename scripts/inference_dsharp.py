import os
import json
import torch 
import numpy as np
from score_models import ScoreModel
import sys
sys.path.append("../src/inference")
from forward_model import complex_to_real, score_likelihood, model
from posterior_sampling import euler_sampler, pc_sampler

device = "cuda" if torch.cuda.is_available() else "cpu"

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args):
    score_model = ScoreModel(checkpoints_directory=args.score_dir)
    img_size = args.img_size
    npix = args.npix
    pad = (npix-img_size)//2

    # Importing data
    data_dir = args.gridded_data_dir    
    with np.load(data_dir) as data_gridded: 
            vis_bin_re = torch.tensor(data_gridded["vis_bin_re"]).to(device)
            vis_bin_imag = torch.tensor(data_gridded["vis_bin_imag"]).to(device)
            std_bin_re = torch.tensor(data_gridded["std_bin_re"]).to(device)
            std_bin_imag = torch.tensor(data_gridded["std_bin_imag"]).to(device)
            counts = torch.tensor(data_gridded["counts"]).to(device)

    # Sampling function 
    S = counts>0

    # Sampled visibilities. 
    vis_sampled = (vis_bin_re + 1j * vis_bin_imag)[S]
    std_sampled = (std_bin_re + 1j * std_bin_imag)[S]

    # Observation and diagonal of the covariance matrix. 
    y = complex_to_real(vis_sampled) * npix
    sigma_y = complex_to_real(std_sampled) * npix

    if "probes" in args.score_dir:
        print(f"Running inference with probes {img_size}*{img_size}...") 
        B, C = 1/2, 1/2 # Check if this is the case for PROBES 256*256. 
        dataset = "probes"
        
    elif "skirt" in args.score_dir: 
        print(f"Running inference with skirt {img_size}*{img_size}...")
        B, C = 1, 0
        dataset = "skirt"

    model_params = (S, B, C, pad)  

    if args.sampler == "euler": 
        print("Sampling with the Euler-Maruyama sampler...")
        sampling_params = [args.pred_steps]
        samples = euler_sampler(
        y = y,
        sigma_y = sigma_y, 
        forward_model = model, 
        score_model = score_model,
        score_likelihood =  score_likelihood, 
        model_parameters = model_params,
        num_samples = args.num_samples,
        num_steps = args.pred_steps,  
        tweedie = args.tweedie, # Experimental 
        keep_chain = False, 
        debug_mode = args.debug_mode, 
        img_size = (args.img_size, args.img_size)
    )

    elif args.sampler == "pc":
        print("Sampling with the Predictor-Corrector sampler...")
        sampling_params = [args.pred_steps, args.corr_steps, args.snr] 
        samples = pc_sampler(
        y = y,
        sigma_y = sigma_y, 
        forward_model = model, 
        score_model = score_model,
        score_likelihood = score_likelihood, 
        model_parameters = model_params,
        num_samples = 1,
        pc_params = sampling_params,  
        tweedie = False, 
        keep_chain = False, 
        debug_mode = False, 
        img_size = (256, 256)
    )
    if args.sanity_plot: 
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize = (8, 4))
        im = axs[1].imshow(samples[0].squeeze().cpu(), cmap = "magma")
        plt.colorbar(im, ax = axs[0], fraction = 0.046)
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
                "model_pixels": img_size,
                "sigma_y": sigma_y,
                "sampler": args.sampler,
                "sampling_params": sampling_params,  # can't save a tuple.
                "num_sims": N_WORKERS,              
                "num_samples": args.num_samples
            }
            filename = "params.json"
            params_dir = os.path.join(path, filename)
            with open(params_dir, "w") as json_file: 
                json.dump(data, json_file, indent = 2)


    else: 
         raise ValueError("Sampler specified does not exist; choose between 'pc' and 'euler'.")      




if __name__ == "__main__":
    ...