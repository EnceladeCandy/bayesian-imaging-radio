from astropy.visualization import ImageNormalize, AsinhStretch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
import mpol.constants as const
import os
from functools import partial
from glob import glob
import json

def grid(pixel_scale, img_size): 
    """Given a pixel scale and a number of pixels in image space, grid the associated Fourier space

    Args:
        pixel_scale (float): Pixel resolution (in arcsec)
        img_size (float/int): Size of the image
    
    Returns:
        edges coordinates of the grid in uv space.     
    """

    # Arcsec to radians: 
    dl = pixel_scale * const.arcsec
    dm = pixel_scale * const.arcsec

    du = 1 / (img_size * dl) * 1e-3 # klambda
    dv = 1 / (img_size * dm) * 1e-3 # klambda

    u_min = -img_size//2 * du 
    u_max =  img_size//2 * du 

    v_min = -img_size//2 * dv
    v_max =  img_size//2 * dv

    u_edges = np.linspace(u_min, u_max, img_size + 1)
    v_edges = np.linspace(v_min, v_max, img_size + 1)

    return u_edges, v_edges


# Defining the window functions, add more in the future:
def pillbox_window(u, center, pixel_size=0.015, m=1):
    """
    u: coordinate of the data points to be aggregated (u or v)
    center: coordinate of the center of the pixel considered. 
    pixel_size: Size of a pixel in the (u,v)-plane, in arcseconds
    m: size of the truncation of this window (in term of pixel_size)
    """
    return np.where(np.abs(u - center) <= m * pixel_size / 2, 1, 0)


def sinc_window(u, center, pixel_size=0.015, m=1):
    """
    u: coordinate of the data points to be aggregated (u or v)
    center: coordinate of the center of the pixel considered. 
    pixel_size: Size of a pixel in the (u,v)-plane, in arcseconds
    m: size of the truncation of this window (in term of pixel_size)
    """
    return np.sinc(np.abs(u - center) / m / pixel_size)

from typing import Callable

def bin_data(u, v, values, weights, bins, window_fn, truncation_radius, statistics_fn="mean", verbose=0):
    """
    u: u-coordinate of the data points to be aggregated
    v: v-coordinate of the data points to be aggregated 
    values: value at the different uv coordinates.
    bins: grid edges
    window_fn: Window function for the convolutional gridding
    truncation_radius:  
    pixel_size: Size of a pixel in the (u,v)-plane, in arcseconds
    m: size of the truncation of this window (in term of pixel_size)
    """
    u_edges = bins[0]
    v_edges = bins[1]
    n_coarse = 0
    grid = np.zeros((len(u_edges)-1, len(v_edges)-1))
    if verbose:
        print("Fitting the KD Tree on the data...")
    # Build a cKDTree from the data points coordinates to query uv points in our truncation radius
    uv_grid = np.vstack((u.ravel(), v.ravel())).T
    tree = cKDTree(uv_grid)
    if verbose:
        print("Gridding...")
    for i in tqdm(range(len(u_edges)-1), disable=not verbose):
        for j in range(len(v_edges)-1):
            # Calculate the coordinates of the center of the current cell in our grid
            u_center = (u_edges[i] + u_edges[i+1])/2
            v_center = (v_edges[j] + v_edges[j+1])/2
            # Query the tree to find the points within the truncation radius of the cell
            indices = tree.query_ball_point([u_center, v_center], truncation_radius, p=1) # p=1 is the Manhattan distance (L1)
            # Apply the convolutional window and weighted averaging
            if len(indices) > 0:
                value = values[indices]
                weight = weights[indices] * window_fn(u[indices], u_center) * window_fn(v[indices], v_center)
                #if len(indices) == 1 and verbose > 1:
                    #print(f"Cell {(i, j)} has a single visibility and weight {weight.sum()} {weight}...")
                if weight.sum() > 0.: # avoid dividing by a normalization = 0
                    if statistics_fn=="mean":
                        grid[j, i] = (value * weight).sum() / weight.sum()
                    elif statistics_fn=="std":
                        m = 1
                        if (weight > 0).sum() < 5:
                            # run statistics on a larger neighborhood
                            while (weight > 0).sum() < 5: # this number is a bit arbitrary, we just hope to get better statistics
                                m += 0.1
                                indices = tree.query_ball_point([u_center, v_center], m*truncation_radius, p=1) # p=1 is the Manhattan distance (L1)
                                value = values[indices]
                                weight = weights[indices] * window_fn(u[indices], u_center, m = m) * window_fn(v[indices], v_center, m = m)
                            #print(f"Coarsened pixel to {m} times its size, now has {len(indices)} for statistics")
                            n_coarse += 1
                        # See https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
                        # See more specifically the bootstrapping
                        if np.sum(weight > 0) < 2:
                            print("Low weight")
                        
                        #N_eff taken from Bevington, see the page: http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
                        importance_weights = window_fn(u[indices], u_center, m = m) * window_fn(v[indices], v_center, m = m)
                        n_eff = np.sum(importance_weights)**2 / np.sum(importance_weights**2)
                        grid[j, i] = np.sqrt(np.cov(value, aweights=weight, ddof = 0)) * (n_eff / (n_eff - 1)) * 1/(np.sqrt(n_eff))
                    elif statistics_fn=="count":
                        grid[j, i] = (weight > 0).sum()
                    elif isinstance(statistics_fn, Callable):
                        grid[j, i] = statistics_fn(value, weight)
    print(f"number of coarsened pix: {n_coarse}")
    return grid

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args):
    npz_dir = args.npz_dir 
    output_dir = args.output_dir # must be a folder
    
    # To grid multiple datasets across nodes (npz_dir must be the path to a folder)
    if N_WORKERS > 1: 
        paths = glob(os.path.join(npz_dir, "*.npz"))
        assert len(paths) > 1
        print(f"ARRAY JOB, THIS_WORKER={THIS_WORKER}")
        path = paths[THIS_WORKER]
        basename = os.path.basename(path)
        target_name, _= os.path.splitext(basename)
        dirname = os.path.dirname(path)
        # Importing target parameters 
        target_params_dir = os.path.join(dirname, "dsharp_params.json")
        with open(target_params_dir, 'r') as file:
            data = json.load(file) 
        pixel_scale, _, _ = data[target_name]

    
    # If gridding only one dataset (npz_dir must be the path to an npz file)
    elif N_WORKERS == 1: 
        paths = [args.npz_dir]
        path = paths[0]
        basename = os.path.basename(path)
        target_name, _= os.path.splitext(basename)
        dirname = os.path.dirname(path)
        target_params_dir = os.path.join(dirname, "dsharp_params.json") # I spent some time creating this file, please use it. 
        with open(target_params_dir, 'r') as file:
            data = json.load(file) 
        pixel_scale, _, _ = data[target_name] # format = (pixel_scale, hshift, wshift) 

    
    print(f"Gridding visibilities for {target_name} with pixel size = {pixel_scale:.2g}...")
    # Importing processed DSHARP data.
    data = np.load(path)
    u = data["uu"]
    v = data["vv"]
    vis = data["data"]
    weight = data["weight"]

    # Hermitian augmentation:
    uu = np.concatenate([u, -u])
    vv = np.concatenate([v, -v])
    vis_re = np.concatenate([vis.real, vis.real])
    vis_imag = np.concatenate([vis.imag, -vis.imag])
    weight_ = np.concatenate([weight, weight])

    print(f"The measurement set contain {len(uu)} data points")
    npix = args.npix
    img_size = args.img_size
    u_edges, v_edges = grid(pixel_scale = pixel_scale, img_size = npix)
    delta_u = u_edges[1] - u_edges[0] # this is delta_u, and we should probably call it that in the future
    truncation_radius = delta_u

    if args.window_function == "sinc": 
        window_fn = partial(sinc_window, pixel_size=delta_u)
    
    elif args.window_function == "pillbox": 
        window_fn = partial(pillbox_window, pixel_size=delta_u)
    else:
        raise ValueError("The window function specified is not implemented yet or does not exist ! Choose between 'sinc' and 'pillbox'.")

        # Real part mean and count
    if not args.debug_mode:
        params = (uu, vv, vis_re, weight_, (u_edges, v_edges), window_fn, truncation_radius)
        vis_bin_re = bin_data(*params, statistics_fn="mean", verbose=1)
        std_bin_re = bin_data(*params, statistics_fn="std", verbose=2)

        # Image part mean
        params = (uu, vv, vis_imag, weight_, (u_edges, v_edges), window_fn, truncation_radius)
        vis_bin_imag = bin_data(*params, statistics_fn="mean", verbose=1)
        std_bin_imag = bin_data(*params, statistics_fn="std", verbose=2)

        # Count: 
        counts = bin_data(*params, statistics_fn="count", verbose=1)
    
    save_dir = os.path.join(output_dir, f"{target_name}_{npix}_gridded_{args.window_function}.npz")
    if not args.debug_mode:
        np.savez(
            save_dir, 
            vis_bin_re = vis_bin_re,
            vis_bin_imag = vis_bin_imag,
            std_bin_re = std_bin_re,
            std_bin_imag = std_bin_imag,
            counts = counts
        )

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Sampling parameters
    parser.add_argument("--npz_dir",            required = True,                                    help = "Path to the processed .npz measurement set")
    parser.add_argument("--output_dir",         required = True,                                    help = "Directory where to save the gridded visibilities (specify a folder not any type of file)")
    # parser.add_argument("--pixel_scale",        required = True, type = float,  default = 0.05,     help = "In arcsec")
    parser.add_argument("--npix",               required = False, type = int,    default = 4096,    help = "Total number of pixels of the padded image.")
    parser.add_argument("--window_function",    required = False, type = str,    default = "sinc",  help = "Either 'sinc' or 'pillbox'")
    parser.add_argument("--img_size",           required = False, type = int,    default = 256,     help = "Number of pixels of the image (must be the same dimension as the dimensions of the score model)")
    parser.add_argument("--debug_mode",         required = False, type = bool,   default = False,   help = "To debug the code, skip the gridding.")
    args = parser.parse_args()
    main(args) 
