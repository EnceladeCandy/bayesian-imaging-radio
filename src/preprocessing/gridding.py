import sys
sys.path.append("../../scripts/")
import torch 
from astropy.visualization import ImageNormalize, AsinhStretch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
import mpol.constants as const

from functools import partial


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
    u_max = img_size//2 * du 

    v_min = -img_size//2 * dv
    v_max = img_size//2 * dv

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