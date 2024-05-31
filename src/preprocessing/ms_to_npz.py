from visread import process, scatter
from casatools import msmetadata
import numpy as np
import os
from glob import glob
from tqdm import tqdm 

#TODO Make this file only one function taking a measurement set directory and creating an npz file.

def main(args):
    fname = args.ms_dir

    if args.recursive: 
        print("Recursive mode; converting all the .ms files of the directory in .npz files.")
        paths = glob(os.path.join(fname, "*.ms"))
        
        for i in tqdm(range(len(paths))):
            print(f"Processing visibilities Measurement set {i+1}/{len(paths)}")
            path = paths[i]
            base_name = os.path.basename(path)
            ms_name, _ = os.path.splitext(base_name)

            # get all spws
            msmd = msmetadata()
            msmd.open(path)
            spws = msmd.datadescids()
            msmd.done()

            uu = []
            vv = []
            weight_ls = []
            data = []
            vis_per_spw = []
            freq_per_spw = []
            sigma_rescale_spw = []
            num_chans = []

            
            print("Iterating over spectral windows...")
            for spw in spws:
                # No rescaling. 
                sigma_rescale=1.

                # Get visibilities
                d = process.get_processed_visibilities(path, spw, sigma_rescale=1.0)
                
                # flatten and concatenate
                flag = d["flag"]
                chan_freq = d["frequencies"] # Hertz
                nchan = len(chan_freq)
                u = d["uu"] # meters
                v = d["vv"] # meters
                
                # Broadcasting shapes so that they are (nchan, N_vis) == flag.shape
                weight = d["weight"] 
                broadcasted_weight = weight * np.ones(shape = (nchan, weight.shape[0]))

                # Convert the uv points to klambdas given the channel frequency
                u, v = process.broadcast_and_convert_baselines(u, v, chan_freq) 
                
                # Applying the flag mask flattens each array:
                uu.append(u[~flag])
                vv.append(v[~flag])
                weight_ls.append(broadcasted_weight[~flag])
                data.append(d["data"][~flag])
                freq_per_spw.append(chan_freq)
                vis_per_spw.append(len(u[~flag]))
                sigma_rescale_spw.append(sigma_rescale)
                num_chans.append(len(chan_freq))
            
            

            print(f"Saving MeasurementSet {ms_name} data into a .npz file...")
            save_dir = os.path.join(args.npz_dir, f"{ms_name}.npz")
            np.savez(
                save_dir,
                uu = np.concatenate(uu),
                vv = np.concatenate(vv),
                weight = np.concatenate(weight_ls),
                data = np.concatenate(data), 
                sigma_rescale_spw = np.array(sigma_rescale_spw),
                vis_per_spw = np.array(vis_per_spw),
                freq_per_spw = np.concatenate(freq_per_spw),
                num_freq_per_spw = np.array(num_chans)
            )

            if args.debug_mode:
                print("Everything working just fine !")
                break

    else: 
        # get all spws
        msmd = msmetadata()
        msmd.open(fname)
        spws = msmd.datadescids()
        msmd.done()

        uu = []
        vv = []
        weight_ls = []
        data = []
        vis_per_spw = []
        freq_per_spw = []
        sigma_rescale_spw = []
        num_chans = []

        # for a given spw
        for spw in spws:
            # No rescaling. 
            sigma_rescale=1.

            # Get visibilities
            d = process.get_processed_visibilities(fname, spw, sigma_rescale=1.0)
            
            # flatten and concatenate
            flag = d["flag"]
            chan_freq = d["frequencies"] # Hertz
            nchan = len(chan_freq)
            u = d["uu"] # meters
            v = d["vv"] # meters
            
            # Broadcasting shapes so that they are (nchan, N_vis) == flag.shape
            weight = d["weight"] 
            broadcasted_weight = weight * np.ones(shape = (nchan, weight.shape[0]))

            # Convert the uv points to klambdas given the channel frequency
            u, v = process.broadcast_and_convert_baselines(u, v, chan_freq) 
            
            # Applying the flag mask flattens each array:
            uu.append(u[~flag])
            vv.append(v[~flag])
            weight_ls.append(broadcasted_weight[~flag])
            data.append(d["data"][~flag])
            freq_per_spw.append(chan_freq)
            vis_per_spw.append(len(u[~flag]))
            sigma_rescale_spw.append(sigma_rescale)
            num_chans.append(len(chan_freq))
        
        
        print("Saving MeasurementSet data into a .npz file...")
        np.savez(
            args.npz_dir,
            uu = np.concatenate(uu),
            vv = np.concatenate(vv),
            weight = np.concatenate(weight_ls),
            data = np.concatenate(data), 
            sigma_rescale_spw = np.array(sigma_rescale_spw),
            vis_per_spw = np.array(vis_per_spw),
            freq_per_spw = np.concatenate(freq_per_spw),
            num_freq_per_spw = np.array(num_chans)
        )

if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Sampling parameters
    parser.add_argument("--ms_dir")
    parser.add_argument("--npz_dir")
    parser.add_argument("--recursive",  type = int,   help = "If 1, activate recursive mode to convert all ms files in a directory; ms_dir and npz_dir arguments must be directory instead of .ms or .npz files")
    parser.add_argument("--debug_mode", type = bool, default = False, help = "Debug mode to skip for loops")
    args = parser.parse_args()
    main(args) 


