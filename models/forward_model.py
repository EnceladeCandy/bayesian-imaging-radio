"""
Author: Noé Dia

This code defines the forward model for the radio interferometric imaging task.
"""

import torch 
import numpy as np
from torch.func import vmap, grad
from posterior_sampling import sigma, mu, complex_to_real


def ft(x): 
    """Compute the orthonormal FFT 2D for x 

    Args:
        x (array): Two-dimensionnal numpy array or torch tensor

    Returns:
        array : FFT 2D of x (computed over the last two dimensions of x, so a batched FFT is possible)
    """
    if type(x) == np.ndarray: 
        return np.fft.fft2(x, norm = "ortho")
    
    if type(x) == torch.Tensor: 
        return torch.fft.fft2(x, norm = "ortho")
    

def ift(x): 
    """Compute the orthonormal FFT 2D for x 

    Args:
        x (array): Two-dimensionnal numpy array or torch tensor

    Returns:
        array : FFT 2D of x (computed over the last two dimensions of x, so a batched FFT is possible)
    """
    if type(x) == np.ndarray: 
        return np.fft.ifft2(x, norm = "ortho")
    
    if type(x) == torch.Tensor: 
        return torch.fft.ifft2(x, norm = "ortho")
    
def noise_padding(x, pad, sigma):
    _, H, W = x.shape
    out = torch.nn.functional.pad(x, (pad, pad, pad, pad)) 
    # Create a mask for padding region
    mask = torch.ones_like(out)
    mask[pad:pad + H, pad:pad+W] = 0.
    
    # Noise pad around the model
    z = torch.randn_like(out) * sigma
    out = out + z * mask
    return out

def model(t, x, score_model, pad, link_function, sampling_function): 
    """Apply a physical model A to a ground-truth x.

    Args:
        t (torch.Tensor): temperature in the sampling procedure of diffusion models.
        x (torch.Tensor): ground-truth 
        score_model (torch.Tensor): trained score-based model (= a prior)
        pad (int): padding to apply to the ground-truth (necessary to have a better resolution in Fourier space)
        + must be chosen so that the padded version of *x* as the same shape as *score_model*
        link_function (torch.Tensor): maps *x* from model space (where the score model was trained) to physical space 
        sampling_function (boolean torch.Tensor): mask selecting the measured visibilities in Fourier space. Must respect sampling_function.shape = (H, W) 

    Returns:
        y_hat (torch.Tensor): 
    """
    x = link_function(x)    
    x_padded = noise_padding(x, pad = pad, sigma = sigma(t, score_model))
    vis_sampled = ft(x_padded).squeeze()[sampling_function] # some troublesome bug makes the squeeze needed here
    y_hat = complex_to_real(vis_sampled)
    return y_hat

def log_likelihood(t, x, y, sigma_y, score_model, forward_model):
    y_hat = forward_model(t, x) # model prediction
    var = sigma(t, score_model) ** 2/2 + mu(t, score_model)**2 * sigma_y ** 2
    var_0 = sigma(t, score_model) ** 2 + mu(t, score_model)**2 * sigma_y ** 2
    diff = (y_hat - mu(t, score_model) * y)**2
    res = diff/var
    res[0] = diff[0]/var_0
    log_prob = - 0.5 * torch.sum(res)
    return log_prob

def score_likelihood(t, x, y, sigma_y, score_model, forward_model): 
    return vmap(grad(lambda t, x: log_likelihood(t, x, y, sigma_y, score_model, forward_model)), randomness = "different")(t, x)