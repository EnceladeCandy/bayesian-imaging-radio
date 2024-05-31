import os
from astropy.visualization import ImageNormalize, AsinhStretch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import mpol.constants as const
from mpol import coordinates
from mpol.gridding import DirtyImager
from functools import partial

"""
Gridding the visibilities using a window function specified
"""


def main(args): 
    # Just take the first spectral window: 
    path = args.ms_dir
    fname = os.path.basename(path)
    target_name, _ = os.path.splitext(fname)
    data = np.load(path)

    u = data["uu"]
    v = data["vv"]
    vis = data["data"]
    weight = data["weight"]
    vis_per_spw = data["vis_per_spw"]

    n_spw = len(vis_per_spw)

    print(f"The dataset contains {2 * len(u)} data points")

    pixel_scale = args.pixel_scale # arcsec
    npix = args.npix # Number of pixels in the grid
    coords = coordinates.GridCoords(cell_size=pixel_scale, npix=npix)
    img_size = args.img_size # Number of pixels in the reconstructed image

    print("Initializing Dirty image")
    imager = DirtyImager(
        coords=coords,
        uu=u,
        vv=v,
        weight=weight,
        data_re=vis.real,
        data_im=vis.imag
    )

    weighting = "briggs"
    robust = args.robust
    print("Accessing dirty image and dirty beam...")
    dirty_image, beam = imager.get_dirty_image(weighting = weighting, robust = robust)

    dirty_image = dirty_image.squeeze()
    beam = beam.squeeze()

    pixel_center = npix // 2
    U_h = pixel_center - args.hshift + img_size//2
    D_h = pixel_center - args.hshift - img_size//2
    U_w = pixel_center - args.wshift + img_size//2
    D_w = pixel_center - args.wshift - img_size//2
    # plt.imshow(dirty_image[D:U, D:U].real, cmap = "magma", origin = "lower")
    # plt.colorbar()
    

    print("Plotting...") 
    fig, axs = plt.subplots(nrows= 1 , ncols = 2, figsize = (7, 3.5))
    axs[0].axis("off")

    axs[0].imshow(beam.real, cmap = "magma", origin = "lower")
    axs[0].set_title("Beam")

    norm = ImageNormalize(dirty_image[..., None, None], vmin = 0, stretch = AsinhStretch())
    im = axs[1].imshow((dirty_image)[D_h:U_h,D_w:U_w].real, cmap =  "magma", norm = norm, origin = "lower")
    axs[1].set(title = "Dirty image", xlabel = r"Right ascension $[\prime\prime]$", ylabel = r"Declination $[\prime\prime]$")
    plt.colorbar(im, fraction = 0.046, label = r"$\frac{Jy}{beam}$")
    plt.subplots_adjust(wspace = 0.5)
    plt.savefig(f"/home/noedia/scratch/dirty_image/{target_name}.jpeg", bbox_inches = "tight", dpi = 300)
    plt.close()

    u_edges = coords.u_edges
    v_edges = coords.v_edges

    def plot_uv_coverage(u, v, grid = True, save = True): 
        # Plot the uv coverage, if grid = True, plot the associated grid
        plt.scatter(u, v, s=0.5, rasterized=True, linewidths=0.0, c="r")

        if grid: 
            for x_coord in u_edges: 
                plt.axvline(x = x_coord, color = "gray", linestyle = "-", alpha = 0.3)

            for y_coord in v_edges:
                plt.axhline(y =y_coord, color = "gray", linestyle = "-", alpha = 0.3)
            

        plt.xlim([u.min(), u.max()])
        plt.ylim([v.min(), v.max()])
        plt.gca().set_aspect("equal")
        if save: 
            plt.savefig(f"/home/noedia/scratch/dirty_image/uv_cov_{target_name}.jpeg", bbox_inches = "tight")
            plt.close()
    
    plot_uv_coverage(u, v, grid = True, save = True)
    
    
if __name__ == "__main__": 
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Sampling parameters
    parser.add_argument("--ms_dir",             required = True,                   help = "Directory to the processed .npz measurement set")
    parser.add_argument("--pixel_scale",        required = True,    type = float,  help = "In arcsec")
    parser.add_argument("--npix",               required = False,    type = int,     default = 4096)
    parser.add_argument("--img_size",           required = False,    type = int,     default = 256)
    parser.add_argument("--robust",             required = False,   default = 0.5,   type = float)
    parser.add_argument("--hshift", required = False, type = int, default = 0)
    parser.add_argument("--wshift", required = False, type = int, default = 0)
    args = parser.parse_args()
    main(args) 