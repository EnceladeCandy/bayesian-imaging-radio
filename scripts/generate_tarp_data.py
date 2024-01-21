import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
from glob import glob
import h5py
from tqdm import tqdm
import scienceplots
from tarp_perso import bootstrapping, get_drp_coverage, mean_coverage
import matplotlib.pylab as pylab

sys.path.append("../models")
from forward_model import link_function

# plt.style.use("science") # Need SciencePLots
# params = {
#          'axes.labelsize': 40,
#          'axes.titlesize': 20,
#          'ytick.labelsize' : 15,
#          'xtick.labelsize' : 15,
#          'xtick.major.size': 4,
#          'xtick.minor.size': 4,
#          'xtick.major.width': 0.5,
#          'xtick.minor.width': 0.5,
#          'ytick.color': "k",
#          'xtick.color': "k",
#          'axes.labelcolor': "k",
#          'ytick.labelcolor' : "k",
#          'xtick.labelcolor' : "k",
#          }
# pylab.rcParams.update(params)

plt.rcParams["font.size"] = 20

def create_dir(dir): 
    if not os.path.exists(dir): 
        os.makedirs(dir)

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))


def main(args):

    # Importing the dataset
    experiment_name = args.experiment_name
    img_size = args.img_size
    samples_dir = args.samples_dir

    if "euler" in samples_dir: 
        sampler = "euler"  
    else: 
        sampler = "pc"
        
    if "probes" in experiment_name:
        B, C = 1/2, 1/2
        sde = "VE PROBES"
        print("Running with PROBES samples")
    else: 
        B, C = 1, 0
        sde = "VP SKIRT"
        print("Running with SKIRT samples")
    
    pattern = experiment_name + "*.h5"
    paths = glob(samples_dir + pattern)

    assert len(paths)>0, "The indicated samples directory does not include any file respecting the experiment name specified."
    
    
    if N_WORKERS>1: 
        num_samples = THIS_WORKER * 100
        print(f"Running the experiment with {num_samples} samples per simulation.")
    else: 
        num_samples = args.num_samples  
    num_sims = len(paths)
    num_dims = args.img_size ** 2

    # Posterior samples and ground-truths 
    samples = np.empty(shape = (num_samples, num_sims, num_dims)) # (n_samples, n_sims, n_dims)
    theta = np.empty(shape = (num_sims, num_dims)) # (n_sims, n_dims)
    
    print("Importing the samples and the ground-truths...")
    num_corrupted_files = 0
    idx_corrupted_files = []
    for i, path in tqdm(enumerate(paths)):
        try:
            with h5py.File(path, "r") as hf:
                hf.keys()            
                samples[:, i, :] = link_function(np.array(hf["model"])[:num_samples].reshape(num_samples, num_dims), B = B, C= C)
                theta[i, :] = np.array(hf["ground_truth"])[:num_samples].flatten()
                hf.close()
        except OSError:
            num_corrupted_files +=1
            idx_corrupted_files.append(i)
        # if args.debug_mode: 
        #     break

    if num_corrupted_files != 0: 
        print(f"Detected {num_corrupted_files} corrupted file(s):")
        for i in range(len(idx_corrupted_files)): 
            print(f"{experiment_name}_{idx_corrupted_files[i]}")
        
        # Removing the corrupted files from the arrays
        samples = np.delete(samples, idx_corrupted_files, axis = 1)
        theta = np.delete(theta, idx_corrupted_files, axis = 0)


    # Sanity check plot: 
    fig, axs = plt.subplots(1, 5, figsize = (5*3.5, 3.5))
    for i in range(len(axs)):
        axs[i].axis("off")
    
    k = np.random.randint(num_sims)
    im = axs[0].imshow(theta[k].reshape(img_size, img_size), cmap = "magma")
    plt.colorbar(im, fraction = 0.046, ax = axs[0])
    for i in range(1, 5):
        im = axs[i].imshow(samples[i, k].reshape(img_size, img_size), cmap = "magma")
        plt.colorbar(im, fraction = 0.046, ax = axs[i])
   
    results_dir = args.results_dir

    # Creating directory for the sanity plot
    save_dir = results_dir + "sanity_check"
    create_dir(save_dir)

    file_name = f"/{sampler}_{experiment_name}.jpeg"
    plt.savefig(save_dir + file_name, bbox_inches="tight", pad_inches=0.2)

    # Creating the second directory if needed 
    save_dir = results_dir + "coverage_data"
    create_dir(save_dir)
    print("Running the tarp test...")
    
    
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
    

    np.savez(save_dir + f"/tarp_{args.method}_{sampler}_{experiment_name}_{num_sims}sims_{num_samples}samples.npz",
             params = {"experiment_name": args.experiment_name, "sde": sde, "sampler" : sampler, "method" : args.method, "num_sims": num_sims, "num_samples" : num_samples,  "img_size": img_size, }, 
             ecp = ecp, 
             ecp_std = ecp_std, 
             alpha = alpha)
    
    # print("Generating the coverage figure...")
    # labels = [0., 0.2, 0.4, 0.6, 0.8, 1]
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi = 150)
    # ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
    # ax.plot(alpha, ecp, label='DRP', color = "red")

    # if args.uncertainty: 
    #     k = args.uncertainty # uncertainty interval = k * sigma
    #     ax.fill_between(alpha, ecp - k * ecp_std, ecp + k * ecp_std, alpha = 0.5, color = "red", label = "99.7\% CI")

    # ax.legend()
    # ax.set_xlabel("Credibility Level")
    # ax.set_ylabel("Expected Coverage Probability")
    # ax.set_xticks(labels)
    # ax.set_xticklabels(labels)
    # ax.set_yticks(labels[1:])
    # ax.set_yticklabels(labels[1:])

    # ax.set(xlim = [0, 1], ylim = [0, 1])
    
    # plt.savefig(save_dir +  f"/{args.method}_{sampler}_{args.experiment_name}_{num_samples}samples{args.output_name}.pdf", bbox_inches = "tight", pad_inches = 0.2)
    # if args.title:
    #     title += rf"${img_size}\times{img_size}$ {num_samples} samples"
    #     plt.title(title, fontsize = 17)
    #     plt.savefig(save_dir + f"/{args.method}_{sampler}_{args.experiment_name}_{num_samples}samples_with_title{args.output_name}.pdf")    



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--method",         required = False,  type = str,   default = False,       help = "Method to use for the TARP figure ('bootstrapping', 'default', 'mean_coverage')")
    parser.add_argument("--experiment_name",required = True,   type = str,                          help = "Same experiment name as the one used in inference_sim.py to generate the samples")
    parser.add_argument("--output_name",    required = False,  type = str,                          help = "Optional. Marker at the end of the plot file to distinguish multiple tarp experiments")
    parser.add_argument("--img_size",       required = True,   type = int)
    parser.add_argument("--num_samples",    required = True,   type = int,                          help = "Number of posterior samples per simulation")
    parser.add_argument("--samples_dir",    required = True,   type = str,                          help = "Directory of the posterior samples")
    parser.add_argument("--results_dir",    required = True,   type = str,                          help = "Directory where the generated plots will be saved. Automatically creates one folder 'sanity_check' and another folder 'coverage_plot' in this directory")
    parser.add_argument("--num_points",     required = False,  type = int,   default = 50,          help = "Number of points in the coverage figure")
    parser.add_argument("--debug_mode",     required = False,  type = bool,  default = False,       help = "Activate debug mode")
    args = parser.parse_args()
    main(args) 
