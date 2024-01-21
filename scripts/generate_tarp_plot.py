import matplotli.pyplot as plt
import matplotlib.pylab as pylab
import scienceplots
import numpy as np

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
    data_dir = args.data_dir  

    # Loading the data
    data = np.load(data_dir)
    params = data["params"]
    alpha = data["alpha"]
    experiment_name, sde, sampler, method,  num_sims, num_samples, img_size = params.values()
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
    
    save_dir = args.results_dir
    plt.savefig(save_dir +  f"/{method}_{sampler}_{args.experiment_name}_{num_samples}samples{args.output_name}.pdf", bbox_inches = "tight", pad_inches = 0.2)
    if args.title:
        title = method + " " + sde + rf" ${img_size}\times{img_size}$ {num_samples} samples"
        plt.title(title, fontsize = 17)
        plt.savefig(save_dir + f"/{args.method}_{sampler}_{args.experiment_name}_{num_samples}samples_with_title{args.output_name}.pdf")    



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser() 
    parser.add_argument("--data_dir",       required = True,    type = str,     help = "Directory of the generated data for the TARP test")
    parser.add_argument("--uncertainty",    required = False,  type = float, default = 0,           help = "Size of the uncertainty zone in the plot")
    parser.add_argument("--results_dir",    required = True,    type = str,     help = "Directory where to save the figure")