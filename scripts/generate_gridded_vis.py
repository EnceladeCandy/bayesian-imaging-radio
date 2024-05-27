import sys
sys.path.append("../../src/")
import torch 
from astropy.visualization import ImageNormalize, AsinhStretch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
import mpol.constants as const

from src.preprocessing.gridding import grid, pillbox_window, sinc_window, bin_data

from functools import partial

"""
Gridding the visibilities using a window function specified
"""


def main(args): 
    # Just take the first spectral window: 
    path = args.ms_dir
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

    pixel_scale = args.pixel_scale
    img_size = args.img_size
    pad = (args.img_padded_size - img_size)//2 
    
    npix = img_size + 2 * pad
    u_edges, v_edges = grid(pixel_scale = pixel_scale, img_size = npix)
    pixel_size = u_edges[1] - u_edges[0] # this is delta_u, and we should probably call it that in the future. 
    
    truncation_radius = pixel_size

    if args.window_function == "sinc": 
        window_fn = partial(sinc_window, pixel_size=pixel_size)
    
    elif args.window_function == "pillbox": 
        window_fn = partial(pillbox_window, pixel_size=pixel_size)

    # Real part mean and count
    params = (uu, vv, vis_re, weight_, (u_edges, v_edges), window_fn, truncation_radius)
    vis_bin_re = bin_data(*params, statistics_fn="mean", verbose=1)
    std_bin_re = bin_data(*params, statistics_fn="std", verbose=2)

    # Image part mean
    params = (uu, vv, vis_imag, weight_, (u_edges, v_edges), window_fn, truncation_radius)
    vis_bin_imag = bin_data(*params, statistics_fn="mean", verbose=1)
    std_bin_imag = bin_data(*params, statistics_fn="std", verbose=2)

    # Count: 
    counts = bin_data(*params, statistics_fn="count", verbose=1)

    np.savez(
        args.npz_dir,
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
    parser.add_argument("--ms_dir",             required = True, help = "Directory to the processed .npz measurement set")
    parser.add_argument("--npz_dir",            required = True, help = "Directory where to save the gridded visibilities (call it ms_gridded.npz)")
    parser.add_argument("--pixel_scale",        required = True, type = float,  help = "In arcsec")
    parser.add_argument("--img_padded_size",    required = True, type = int,     default = 4096)
    parser.add_argument("--window_function",    required = True, help = "Either sinc or pillbox")
    parser.add_argument("--experiment_name")
    parser.add_argument("--img_size",           required = True,    type = int)
    

    args = parser.parse_args()
    main(args) 