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
    
    elif type(x) == torch.Tensor: 
        return torch.fft.fft2(x, norm = "ortho")
    

def ift(x): 
    """Compute the orthonormal FFT 2D for x 

    Args:
        x (array): Two-dimensionnal numpy array or torch tensor

    Returns:
        array : orthonormal FFT 2D of x (computed over the last two dimensions of x, so a batched FFT is possible)
    """
    if type(x) == np.ndarray: 
        return np.fft.ifft2(x, norm = "ortho")
    
    if type(x) == torch.Tensor: 
        return torch.fft.ifft2(x, norm = "ortho")
    

def link_function(x, B, C): 
    """Mapping from the model space (where the score model is trained) to the image space 
    (with meaningful physical units)

    Args:
        x (array, float...): image to map 
        B (float): factor
        C (float): constant

    Returns:
        x in image space
    """
    return B*x + C

def noise_padding(x, pad, sigma):
    """Padding with realizations of noise of the same temperature of the current diffusion step

    Args:
        x (torch.Tensor): ground-truth 
        pad (int): amount of pixels needed to pad x
        sigma (torch.Tensor): std of the gaussian distribution for the noise pad around the model

    Returns:
        out (torch.Tensor): noise padded version of the input
    """

    # To manage batched input
    if len(x.shape)<4:
        _, H, W = x.shape
    else: 
        _, _, H, W = x.shape
    out = torch.nn.functional.pad(x, (pad, pad, pad, pad)) 
    # Create a mask for padding region
    mask = torch.ones_like(out)
    mask[pad:pad + H, pad:pad+W] = 0.
    
    # Noise pad around the model
    z = torch.randn_like(out) * sigma
    out = out + z * mask
    return out

def model(t, x, score_model, model_parameters): 
    """Apply a physical model A to a ground-truth x.

    Args:
        t (torch.Tensor): temperature in the sampling procedure of diffusion models.
        x (torch.Tensor): ground-truth 
        score_model (torch.Tensor): trained score-based model (= the score of a prior)
        model_parameters (Tuple): list of parameters for the model (sampling_function, B, C)
          - index 0: sampling function (mask selecting the measured visibilities in Fourier space, must have a shape (H, W) where H and W are the height
            and width of the padded image respectively)
          - index 1 and index 2: B and C, the link_function parameters (see function link_function)

    Returns:
        y_hat (torch.Tensor): 
    """

    sampling_function, B, C, pad= model_parameters
    x = link_function(x, B, C)    
    x_padded = noise_padding(x, pad = pad, sigma = sigma(t, score_model))
    vis_sampled = ft(x_padded).squeeze()[sampling_function] # some troublesome bug makes the squeeze needed here
    y_hat = complex_to_real(vis_sampled)
    return y_hat


def model_to_plot(t, x, score_model, model_parameters):
    """
    Same function as model(*args) except that the sampling function is replaced by an integer matrix where each pixel can either take
    the value 0 or 1. This 
    Args:
        t (_type_): _description_
        x (_type_): _description_
        score_model (_type_): _description_
        model_parameters (_type_): _description_

    Returns:
        _type_: _description_
    """
    sampling_function, B, C, pad  = model_parameters
    x = link_function(x, B= B, C = C)  
    x_padded = noise_padding(x, pad = pad, sigma = sigma(t, score_model))
    vis_sampled = ft(x_padded) * sampling_function.to(torch.uint8)
    return torch.cat([vis_sampled.real, vis_sampled.imag])


def log_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters):
    """
    Compute the log-likelihood following the convolved likelihood approximation 
    (see Appendix A of https://arxiv.org/abs/2311.18012)

    Args:
        t (torch.Tensor): temperature in the sampling procedure of diffusion models
        x (torch.Tensor): ground-truth
        y (torch.Tensor): observation 
        sigma_y (float): std of the 
        score_model (function): trained score-based model (= the score of a prior)
        forward_model (function): physical model encoding the measurement process of a radio interferometer
        (up to some approximations)

    Returns:
        log-likelihood of x for an observation y
    """

    # Model prediction
    y_hat = forward_model(t, x, score_model, model_parameters) # model prediction
    
    # Variance computed analytically with the convolved likelihood approximation
    var = sigma(t, score_model) ** 2/2 + mu(t, score_model)**2 * sigma_y ** 2
    var_0 = sigma(t, score_model) ** 2 + mu(t, score_model)**2 * sigma_y ** 2
    diff = (y_hat - mu(t, score_model) * y)**2 / var
    diff[0] = (y_hat[0] - mu(t, score_model) * y[0])**2 / var_0 
    

    # Log probability
    log_prob = - 0.5 * torch.sum(diff)
    return log_prob

def score_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters): 
    """
    Compute the score of the likelihood
    See log_likelihood(*args) for a description of the arguments 
    """ 
    return vmap(grad(lambda x, t: log_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters)), randomness = "different")(x, t) # Do not interchange x and t

