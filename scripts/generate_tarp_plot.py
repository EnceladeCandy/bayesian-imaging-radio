import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scienceplots
import json
from utils import create_dir, open_json

plt.style.use("science") # Need SciencePLots
params = {
         'axes.labelsize': 40,
         'axes.titlesize': 20,
         'ytick.labelsize' : 15,
         'xtick.labelsize' : 15,
         'xtick.major.size': 4,
         'xtick.minor.size': 4,
         'xtick.major.width': 0.5,
         'xtick.minor.width': 0.5,
         'ytick.color': "k",
         'xtick.color': "k",
         'axes.labelcolor': "k",
         'ytick.labelcolor' : "k",
         'xtick.labelcolor' : "k",
         }
pylab.rcParams.update(params)



def main(args):

    experiment_name = args.experiment_name
    experiment_dir = args.experiment_dir
    results_dir = args.results_dir
    
    # Importing the experiment's parameters
    json_dir = os.path.join(experiment_dir, experiment_name + ".json")
        
    # Loading the experiment's parameters
    params = open_json(json_dir)

    dataset = params["dataset"]
    sde = params["sde"]
    sampler = params["sampler"]
    num_sims = params["num_sims"]
    num_samples = params["num_samples"]
    img_size = params["model_pixels"]
    sampling_params = params["sampling_params"] # Format = (Predictor, Corrector, SNR) for pc sampler | (Predictor) for euler sampler
    sigma_y = params["sigma_y"]
       
    # Following the filename in generate_tarp_data.py
    # if sampler.lower() == 'pc' or sampler.lower() == "old_pc": 
    #     predictor = sampling_params[0]
    #     corrector = sampling_params[1]
    #     snr = sampling_params[-1]
    #     data_dir = f"/tarp_{args.method}_{experiment_name}_{sampler}_{predictor}pred_{corrector}corr_{snr:.1e}.npz"

    # elif sampler.lower() == "euler": 
    #     num_steps = sampling_params[0]
    #     data_dir = f"/tarp_{args.method}_{experiment_name}_{sampler}_{predictor}steps.npz"
 
    # data_dir = os.path.join(args.experiment_dir, f"coverage_data/{data_dir}")
    data_dir = args.data_dir
    data = np.load(data_dir)
    alpha = data["alpha"]
    ecp = data["ecp"]
    ecp_std = data["ecp_std"]

    print("Generating the coverage figure...")
    labels = [0., 0.2, 0.4, 0.6, 0.8, 1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi = 150)
    ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
    ax.plot(alpha, ecp, label='DRP', color = "red")

    if args.uncertainty: 
        k = args.uncertainty # uncertainty interval = k * ecp_std
        ax.fill_between(alpha, ecp - k * ecp_std, ecp + k * ecp_std, alpha = 0.5, color = "red", label = "99.7\% CI")

    ax.legend()
    ax.set_xlabel("Credibility Level")
    ax.set_ylabel("Expected Coverage Probability")
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.set_yticks(labels[1:])
    ax.set_yticklabels(labels[1:])
    ax.set(xlim = [0, 1], ylim = [0, 1])

    # Creating folder for the plots and saving: 
    save_dir = os.path.join(args.results_dir, "coverage_plot")
    create_dir(save_dir)
    
    
    if sampler.lower() == "pc" or  sampler.lower() == "old_pc": 
        predictor, corrector, snr = sampling_params
        figure_file = f"{args.method}_{sampler}_{predictor}pred_{corrector}corr_{snr}snr.pdf"
        save_dir = os.path.join(save_dir, figure_file)

    elif sampler.lower() == "euler": 
        num_steps = sampling_params
        figure_file = f"{args.method}_{sampler}_{num_steps}steps.pdf"
        save_dir = os.path.join(save_dir, figure_file)
    
    else: 
        raise ValueError("The sampler specified in the .json file does not exist or isn't supported")

    plt.savefig(save_dir, bbox_inches = "tight", pad_inches = 0.2)

    # if args.title:
    #     title = method + " " + sde + " " + dataset + rf" ${img_size}\times{img_size}$ {num_samples} samples"
    #     plt.title(title, fontsize = 17)
    #     plt.savefig(save_dir2)    



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser() 
    parser.add_argument("--experiment_dir",       required = True)
    parser.add_argument("--data_dir",       required = True)
    parser.add_argument("--experiment_name",          required = True)
    parser.add_argument("--uncertainty",             required = False,  type = float, default = 0,           help = "Size of the uncertainty zone in the plot")
    parser.add_argument("--results_dir",             required = True,    type = str,     help = "Directory where to save the figure")
    parser.add_argument("--method",                 required = True,    type = str)
    args = parser.parse_args()
    main(args)