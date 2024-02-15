
import os 
import sys
sys.path.append("../src/data_analysis")
sys.path.append("../src/inference")
import numpy as np
from tqdm import tqdm
from glob import glob
from utils import load_json, create_dir
from chi_squared_fit import fit_chi2
from forward_model import link_function
import datetime




# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args):
    if args.grid == True: 
        predictor = args.predictor
        corrector = args.corrector 
        snr = args.snr
        experiment_dir = os.path.join(args.experiment_dir, f"{predictor}pred_{corrector}corr_{snr}snr")
        sampling_params = np.array([predictor, corrector, snr])

    else: 
        experiment_dir = args.experiment_dir

    # Importing the experiment's parameters
    params_dir = os.path.join(experiment_dir, "params.json")
    params = load_json(params_dir)
    experiment_name = params["experiment_name"]
    sampler = params["sampler"]
    num_sims = params["num_sims"]
    num_samples = params["num_samples"]
    img_size = params["model_pixels"]
    sigma_y = params["sigma_y"]
    sampling_params = np.array(params["sampling_params"]) # Format = (Predictor, Corrector, SNR) for pc sampler | (Predictor) for euler sampler

    # Loading the paths for each posterior (1 path = 1 simulation = 1 observation = 1 posterior sampled num_samples times)
    pattern = "samples_sim_*.npz"
    data_dir = os.path.join(experiment_dir, pattern)
    paths = glob(data_dir)
    assert len(paths)>0, "The indicated samples directory does not include any file respecting the experiment name specified."

    print("Importing the observations and the reconstructions to compute chi-squared samples...")
    idx_corrupted_files = []
    chi_samples = np.empty(shape = (num_sims, num_samples))

    for i, path in tqdm(enumerate(paths)):
        try:
            data = np.load(path)
            y = data["observation"]
            y_hat = data["reconstruction"]
            chi_samples[i] = np.sum((y - y_hat) ** 2 / sigma_y ** 2, axis = 1)
        except OSError:    
            idx_corrupted_files.append(i)
        
        # Exit the loop if testing the code
        if args.debug_mode and i==20: 
            break

    if len(idx_corrupted_files) > 0: 
        print(f"Detected {len(idx_corrupted_files)} corrupted file(s):")
        for i in range(len(idx_corrupted_files)): 
            print(f"{experiment_name}_{idx_corrupted_files[i]}")
        chi_samples = np.delete(chi_samples, idx_corrupted_files) # deleting corrupted data
    


    chi_samples = chi_samples.flatten()
    # # Fitting the number of degrees of freedom k of a chi squared distribution on the observed data
    # result_fit = fit_chi2(k0 = len(y)-10, x = chi_samples) # the number of degrees of freedom should be the dimension of y (i.e. the number of gaussian rvs added)
    # k_star = result_fit.x[0]
    # sigma_k = np.sqrt(result_fit.hess_inv)
    # print(k_star)
    # print(sigma_k)

    # Creating directory for the data
    save_dir = os.path.join(args.results_dir, "chi_data")
    create_dir(save_dir)

    # # Name for the file 
    # if sampler.lower() == 'pc' or sampler.lower() == "old_pc": 
    #     predictor = sampling_params[0]
    #     corrector = sampling_params[1]
    #     snr = sampling_params[-1]
    #     filename = f"chi_samples_{sampler}_{THIS_WORKER}.npz"
    #     filedir = os.path.join(save_dir, filename)

    # elif sampler.lower() == "euler": 
    #     num_steps = sampling_params[0]
    #     filename = f"fitted_chi_{num_steps}steps.npz"
    #     filedir = os.path.join(save_dir, filename)

    
    # Get the current time
    now = datetime.datetime.now()

    # Format the time as a string
    time_string = now.strftime("%Y-%m-%d_%H:%M:%S")
    filename = f"chi_samples_{sampler}_{time_string}.npz"
    filedir = os.path.join(save_dir, filename)
    
    # Saving 
    np.savez(filedir,
             chi_samples = chi_samples, 
             sampling_params = sampling_params)


if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Modes (changes how the script is going to run)
    parser.add_argument("--grid",               required = False,  type = bool,  default = False,       help = "To combine with the mode argument if you want to run multiple TARP tests at once. You then need to specify predictor, corrector and snr.")
    
    # Input data directory (= directory where the output of inference_sim.py has been saved)
    parser.add_argument("--experiment_dir",        required = True,   type = str,                          help = "Directory of the posterior samples")

    # PC parameters: 
    parser.add_argument("--predictor",          required = False,   default = 1000,     type = int,     help = "Number of steps if sampler is 'euler'. Number of predictor steps if the sampler is 'pc'")
    parser.add_argument("--corrector",          required = False,   default = 20,       type = int,     help = "Number of corrector steps for the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2,     type = float,   help = "Parameter pc sampling")

    # Output directory for the results
    parser.add_argument("--results_dir",        required = True,   type = str,                          help = "Directory where a .npz file containing the results of the TARP test will be saved. A folder 'coverage_data' will be automatically created in this directory")
    
    # Debug mode and Sanity plot
    parser.add_argument("--debug_mode",         required = False,  type = bool,  default = False,       help = "Activate debug mode (leave empty to run without debug_mode)")
    args = parser.parse_args()
    main(args) 

