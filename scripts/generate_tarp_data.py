import os
import matplotlib.pyplot as plt
import numpy as np 
from glob import glob
import json
from tqdm import tqdm
from tarp_perso import bootstrapping, get_drp_coverage, mean_coverage
from utils import create_dir, load_json



# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

"""
File format: 
scratch
    -tarp_experiment
        -gridsearch
            -post_sampling_cl
                -veprobes64
                    -4000pred_1corr_0.1snr
                    -4000pred_1corr_0.01snr

                -vpskirt64
"""

def main(args):
    if args.grid == True: 
        predictor = args.predictor
        corrector = args.corrector 
        snr = args.snr
        experiment_dir = os.path.join(args.experiment_dir, f"{predictor}pred_{corrector}corr_{corrector}_{snr}snr")

    else: 
        experiment_dir = args.experiment_dir


    if args.mode == "posterior": 
        params_dir = os.path.join(experiment_dir, "params.json")
    
        # Importing the experiment's parameters (a bit useless but thought it was more straightforward than specifying at the end of the file)
        params = load_json(params_dir)
        experiment_name = params["experiment_name"]
        sampler = params["sampler"]
        num_sims = params["num_sims"]
        num_samples = params["num_samples"]
        img_size = params["model_pixels"]
        sampling_params = params["sampling_params"] # Format = (Predictor, Corrector, SNR) for pc sampler | (Predictor) for euler sampler

        # Loading the paths for each posterior (1 path = 1 simulation = 1 observation = 1 posterior sampled num_samples times)
        pattern = "samples_sim*.npz"
        data_dir = os.path.join(experiment_dir, sampler, pattern)
        print(data_dir)
        paths = glob(data_dir)
        assert len(paths)>0, "The indicated samples directory does not include any file respecting the experiment name specified."

        # Posterior samples and ground-truths 
        samples = np.empty(shape = (num_samples, num_sims, img_size ** 2)) # (n_samples, n_sims, n_dims)
        theta = np.empty(shape = (num_sims, img_size ** 2)) # (n_sims, n_dims)

        print("Importing the samples and the ground-truths...")
        idx_corrupted_files = []
        for i, path in tqdm(enumerate(paths)):
            try:
                # Loading the samples and the ground-truth 
                data = np.load(path)
                samples[:, i, :] = data["samples"].reshape(-1, img_size ** 2)
                theta[i, :] = data["ground_truth"].reshape(-1, img_size ** 2)

            # To handle bugs that may occur during the array job. 
            except OSError:
                idx_corrupted_files.append(i)
            
            # Exit the loop if testing the code
            if args.debug_mode: 
                break

    elif args.mode == "prior": 
        pattern = "prior_samples*"
        data_dir = os.path.join(args.experiment_dir, sampler, pattern)
        samples = np.empty(shape = (num_sims, num_samples))
        for i, path in tqdm(enumerate(paths)):
            try:
                # Loading the samples and the ground-truth 
                data = np.load(path)
                samples[:, i, :] = data["samples"].reshape(-1, img_size ** 2)
                theta[i, :] = data["ground_truth"].reshape(-1, img_size ** 2)

            # To handle bugs that may occur during the array job. 
            except OSError:
                idx_corrupted_files.append(i)

            if args.debug_mode: 
                break
    
    else: 
        raise ValueError("mode argument must be either 'posterior' or 'prior'.")
    


    # Print a message if corrupted files are detected
    if len(idx_corrupted_files) > 0: 
        print(f"Detected {len(idx_corrupted_files)} corrupted file(s):")
        for i in range(len(idx_corrupted_files)): 
            print(f"{experiment_name}_{idx_corrupted_files[i]}")
        
        # Removing the corrupted files from the arrays
        samples = np.delete(samples, idx_corrupted_files, axis = 1)
        theta = np.delete(theta, idx_corrupted_files, axis = 0)


    # Plotting posterior samples from a randomly selected simulation
    if args.sanity_plot:  
        fig, axs = plt.subplots(1, 5, figsize = (5*3.5, 3.5))
        
        
        im = axs[0].imshow(theta[0].reshape(img_size, img_size), cmap = "magma")
        plt.colorbar(im, fraction = 0.046, ax = axs[0])
        for i in range(1, 5):
            im = axs[i].imshow(samples[0, 0].reshape(img_size, img_size), cmap = "magma")
            plt.colorbar(im, fraction = 0.046, ax = axs[i])
            axs[i].axis("off")

        # Creating directory + saving sanity plot
        save_dir = os.path.join(args.results_dir, "sanity_check")
        create_dir(save_dir)
        file_name = f"{sampler}_{experiment_name}.jpeg"
        file_dir = os.path.join(save_dir, file_name)
        plt.savefig("test.jpeg", bbox_inches="tight", pad_inches=0.2)
    
    

    # Running the TARP test
    if args.method.lower() == "default":
        print("Running a regular tarp test")
        ecp, alpha = get_drp_coverage(samples, theta, references = "random", metric = "euclidean", norm = False)
        ecp_std = np.zeros_like(ecp)

    elif args.method.lower() == "bootstrapping": 
        print("Applying the bootstrapping method")
        ecp, ecp_std, alpha = bootstrapping(samples, theta, references = "random", metric = "euclidean", norm = False, debug_mode = args.debug_mode, num_points = args.num_points)

    elif args.method.lower() == "mean_coverage": 
        print("Averaging over N_sims coverage tests")
        ecp, ecp_std, alpha = mean_coverage(samples, theta, references = "random", metric = "euclidean", norm = False, debug_mode = args.debug_mode, num_points = args.num_points)
    else: 
        raise ValueError("The method specified does not exist. Choose between 'default', 'mean' and 'bootstrapping'")
    

    # Creating directory for the data
    save_dir = os.path.join(args.results_dir, "coverage_data")
    create_dir(save_dir)

    # Name for the file 
    if sampler.lower() == 'pc' or sampler.lower() == "old_pc": 
        predictor = sampling_params[0]
        corrector = sampling_params[1]
        snr = sampling_params[-1]
        filename = f"tarp_{args.method}_{predictor}pred_{corrector}corr_{snr}snr.npz"
        filedir = os.path.join(save_dir, filename)

    elif sampler.lower() == "euler": 
        num_steps = sampling_params[0]
        filename = f"tarp_{args.method}_{num_steps}steps.npz"
        filedir = os.path.join(save_dir, filename)

    # Saving 
    np.savez(filedir,
             ecp = ecp, 
             ecp_std = ecp_std, 
             alpha = alpha)
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Input data directory (= directory where the output of inference_sim.py has been saved)
    parser.add_argument("--experiment_dir",        required = True,   type = str,                          help = "Directory of the posterior samples")

    # PC parameters: 
    parser.add_argument("--predictor",          required = False,   default = 1000,     type = int,     help = "Number of steps if sampler is 'euler'. Number of predictor steps if the sampler is 'pc'")
    parser.add_argument("--corrector",          required = False,   default = 20,       type = int,     help = "Number of corrector steps for the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2,     type = float,   help = "Parameter pc sampling")

    # Coverage method and parameters
    parser.add_argument("--method",             required = False,  type = str,   default = False,       help = "Method to use for the TARP figure ('bootstrapping', 'default', 'mean_coverage')")
    parser.add_argument("--num_points",         required = False,  type = int,   default = 50,          help = "Number of points in the coverage figure")
    
    # Experiment name
    parser.add_argument("--experiment_name",    required = True,   type = str,                          help = "Should be the same parameter as the one used in inference_sim.py to generate the posterior samples")

    # Output directory for the results
    parser.add_argument("--results_dir",        required = True,   type = str,                          help = "Directory where a .npz file containing the results of the TARP test will be saved. A folder 'coverage_data' will be automatically created in this directory")
    #parser.add_argument("--output_name",        required = True,   type = str,                          help = "Suffix to add at the end of the file name for the coverage data")
    
    # Code modes
    parser.add_argument("--mode",               required = True,   type = str,                          help = "Running mode for the script. Either 'posterior' or 'prior'. If 'posterior'/'prior' the script will run expecting to receive posterior/prior samples to run the tarp test.")
    parser.add_argument("--grid",               required = False,  type = bool,  default = False,       help = "To combine with the mode argument if you want to run multiple TARP tests at once")
    
    # Debug mode and Sanity plot
    parser.add_argument("--debug_mode",         required = False,  type = bool,  default = False,       help = "Activate debug mode (leave empty to run without debug_mode)")
    parser.add_argument("--sanity_plot",        required = False,  type = bool,  default = False,       help = "Creates a plot with posterior samples and the ground-truth for a random simulation if True. Leave empty to not generate this plot.")
    args = parser.parse_args()
    main(args) 
