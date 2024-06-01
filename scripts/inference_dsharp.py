import os
import sys
import json
import torch 
import numpy as np
from glob import glob 
from score_models import ScoreModel
sys.path.append("../src/inference")
from forward_model import complex_to_real, score_likelihood, model
from utils import create_dir
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
    npz_dir = args.npz_dir 
    output_dir = args.output_dir # must be a folder

    # To grid multiple datasets across nodes (npz_dir must be the path to a folder)
    if N_WORKERS > 1: 
        paths = glob(os.path.join(npz_dir, "*.npz"))
        assert len(paths) > 1
        print(f"ARRAY JOB, THIS_WORKER={THIS_WORKER}")
        path = paths[THIS_WORKER]

    # If gridding only one dataset (npz_dir must be the path to an npz file)
    elif N_WORKERS == 1: 
        path = npz_dir
    
    basename = os.path.basename(path)
    target_name, _= os.path.splitext(basename)
    parts = target_name.rsplit(f"_{args.npix}_gridded_sinc", 1)
    target_name = parts[0] if len(parts) > 1 else target_name
    dirname = os.path.dirname(path)
    # Importing target parameters 
    target_params_dir = os.path.join(dirname, "dsharp_params.json") # a file with the pixel scale info for each target. 
    with open(target_params_dir, 'r') as file:
        data = json.load(file) 
    _, xshift, yshift = data[target_name]  # (pixel_scale, xshift, yshift), shifts to the center of the 256*256 image we want to reconstruct. 
    
    
    # Importing gridded visibilities + noise estimates.  
    print("Importing gridded visibilities...")
    with np.load(path) as data_gridded: 
            vis_bin_re = torch.tensor(data_gridded["vis_bin_re"]).to(device)
            vis_bin_imag = torch.tensor(data_gridded["vis_bin_imag"]).to(device)
            std_bin_re = torch.tensor(data_gridded["std_bin_re"]).to(device)
            std_bin_imag = torch.tensor(data_gridded["std_bin_imag"]).to(device)
            counts = torch.tensor(data_gridded["counts"]).to(device)

    img_size = args.img_size # Number of rows/cols of the target image (in our case 256*256)
    npix = args.npix # Number of rows/cols of the gridded visibilities 
    
    S = counts>0 # sampling function

    # Throwing out empty cells on the Fourier grid.  
    vis_sampled = (vis_bin_re + 1j * vis_bin_imag)[S]
    std_sampled = (std_bin_re + 1j * std_bin_imag)[S]

    # Complex tensors into their vectorized real representations 
    y = complex_to_real(vis_sampled) * npix
    sigma_y = complex_to_real(std_sampled) * npix

    # Irregular padding 
    xc = npix//2 - xshift # center of the 256*256 pixel grid we want to reconstruct
    yc = npix//2 - yshift 

    # Irregular padding
    pad_b = (yc - img_size // 2)
    pad_t = npix - (yc + img_size // 2)
    pad_l = (xc - img_size // 2)
    pad_r = npix - (xc + img_size // 2)
    pad = (pad_l, pad_r, pad_b, pad_t) # torch convention (pad left, pad right, pad bottom, pad top)

    if "probes" in args.score_dir:
        print(f"Running inference with probes {img_size}*{img_size} for target {target_name}...") 
        B, C = 1/2, 1/2 # Check if this is the case for PROBES 256*256. 
        dataset = "probes"
        
    elif "skirt" in args.score_dir: 
        print(f"Running inference with skirt {img_size}*{img_size} for target {target_name}...")
        B, C = 1, 0
        dataset = "skirt"

    padding_mode = args.padding_mode # either 'noise' or 'zero' 
    model_params = (S, B, C, pad, padding_mode)  

    # Generating posterior samples
    num_samples = args.num_samples
    batch_size = args.batch_size
    total_samples = np.empty(shape = [num_samples, 1, img_size, img_size], dtype = np.float32) 
    for i in range(int(num_samples//batch_size)):
        if args.sampler == "euler": 
            print("Sampling with the Euler-Maruyama sampler...")
            samples = euler_sampler(
            y = y,
            sigma_y = sigma_y, 
            forward_model = model, 
            score_model = score_model,
            score_likelihood =  score_likelihood, 
            model_parameters = model_params,
            num_samples = args.num_samples,
            num_steps = args.predictor,  
            tweedie = args.tweedie, # Experimental 
            keep_chain = False, 
            debug_mode = args.debug_mode, 
            img_size = (args.img_size, args.img_size)
        )

        elif args.sampler == "pc":
            print("Sampling with the Predictor-Corrector sampler...")
            sampling_params = [args.predictor, args.corrector, args.snr] 
            samples = pc_sampler(
            y = y,
            sigma_y = sigma_y, 
            forward_model = model, 
            score_model = score_model,
            score_likelihood = score_likelihood, 
            model_parameters = model_params,
            num_samples = 1,
            pc_params = sampling_params,  
            tweedie = args.tweedie, 
            keep_chain = False, 
            debug_mode = args.debug_mode, 
            img_size = (256, 256)
        )

        else: 
            raise ValueError("Sampler specified does not exist; choose between 'pc' and 'euler'.") 
        
        total_samples[i * batch_size: (i+1) * batch_size] = link_function(samples, B, C).cpu().numpy().astype(np.float32)
        if args.debug_mode:
             break
        
    # Creating experiment's directory
    print("Creating folder for the experiment in the results directory...")
    path_target = os.path.join(args.output_dir, target_name)
    create_dir(path_target)

    print("Creating folder for score model used...")
    path_dataset = os.path.join(path_target, dataset)
    
    # Creating sampler directory
    print("Creating folder for the sampler used in the experiment's directory...")
    path_sampler = os.path.join(path_dataset, args.sampler)
    create_dir(path_sampler)

    # Creating directory according to the pc parameter being used
    print("Creating folder for the parameters used in the sampler's directory...")
    if args.sampler == "pc": 
        params_foldername = f"{args.predictor}pred_{args.corrector}corr_{args.snr}snr"
    elif args.sampler == "euler": 
        params_foldername = f"{args.predictor}steps"
    path_params = os.path.join(path_sampler, params_foldername)
    create_dir(path_params)
    
    # Print Saving ! 
    print("Saving posterior samples...")
    samples_dir = os.path.join(path_params, f"{target_name}_{THIS_WORKER}.npy")
    np.save(samples_dir, total_samples)
    print("Posterior samples saved !")
    if args.sanity_plot: 
            import matplotlib.pyplot as plt
            from astropy.visualization import ImageNormalize, AsinhStretch
            norm = ImageNormalize((total_samples[0]/total_samples[0].max()).squeeze().cpu().numpy(), vmin = 0, stretch = AsinhStretch(a=0.05))
            fig, axs = plt.subplots(1, 1, figsize = (8, 4))
            im = axs.imshow((total_samples[0]/total_samples[0].max()).squeeze().cpu(), cmap = "magma", origin = "lower", norm = norm)
            plt.colorbar(im, ax = axs, fraction = 0.046)
            axs.set_title("Posterior sample")

            image_dir = os.path.join(path_params, "sanity.jpeg")
            plt.savefig(image_dir, bbox_inches = "tight")     


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    # Input dir + Output dir
    parser.add_argument("--npz_dir",            required = True,                        type = str,     help = "Path to the gridded visibilities.")
    parser.add_argument("--output_dir",         required = True,                        type = str,     help = "Path to the output folder (a few folders will be created to keep things organized).")
    
    # Data specs. 
    parser.add_argument("--img_size",           required = True,                        type = int,     help = "Image size (only supporting images with equal width and heights for now). The total number of pixels in the image should be img_size * img_size")
    parser.add_argument("--npix",               required = False,   default = 4096,     type = int,     help = "Number of cells in the Fourier grid")
    
    # Sampling parameters
    parser.add_argument("--sampler",            required = False,   default = "euler",  type = str,     help = "Sampler used ('old_pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 10,       type = int,     help = "Total number of posterior samples to generate.")
    parser.add_argument("--batch_size",         required = False,   default = 20,       type = int,     help = "Number of posterior samples to generate per iteration (the code begins a loop if num_samples > batch_size).")
    parser.add_argument("--predictor",          required = False,   default = 4000,     type = int,     help = "Number of steps if sampler is 'euler'. Number of predictor steps if the sampler is 'pc'")
    parser.add_argument("--corrector",          required = False,   default = 1,        type = int,     help = "Number of corrector steps for the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2,     type = float,   help = "Snr parameter for PC sampling")
    parser.add_argument("--score_dir",          required = True,                        type = str,     help = "Path to the trained score model." )
    parser.add_argument("--tweedie",            required = False,   default = False,    type = bool,      help = "Sampler used ('old_pc' or 'euler')")
    parser.add_argument("--padding_mode",       required = False,   default = "zero",   type = str,     help = "Sampler used ('old_pc' or 'euler')")

    
    # For debugging or checking the results. 
    parser.add_argument("--debug_mode",         required = False,   default = False,    type = bool,    help = "True to skip loops and debug")
    parser.add_argument("--sanity_plot",        required = False,   default = False,    type = bool,    help = "True to create a plot with posterior samples and the ground truth (if testing the script put both debug_mode and sanity_plot to True)")

    
    args = parser.parse_args()
    main(args) 