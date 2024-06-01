import torch 
import numpy as np
from torch.func import vmap, grad
import sys 
sys.path.append("../inference")
from posterior_sampling import sigma, mu, complex_to_real

# SOME UTILITY FUNCTIONS FOR THE FORWARD MODEL
def ft(x): 
    """Compute the orthonormal FFT 2D for x.
    Note: In torch's convention, the DC component is expected to be at (0,0) before being passed to the fft. 

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
    """Compute the orthonormal IFT 2D for x. 
    Note: In torch's convention, the DC component is expected to be at (N/2, N/2) (i.e. the center of the image) before being passed to the ift. 

    Args:
        x (array): Two-dimensionnal numpy array or torch tensor

    Returns:
        array : orthonormal FFT 2D of x (computed over the last two dimensions of x, so a batched FFT is possible)
    """
    if type(x) == np.ndarray: 
        return np.fft.ifft2(x, norm = "ortho")
    
    if type(x) == torch.Tensor: 
        return torch.fft.ifft2(x, norm = "ortho")
    
def ftshift(x):
    """
    Places the DC component of the input at the Nyquist Frequency (i.e. the center of the image for a square image). 
    Note: For even length inputs, fftshift and iftshift are equivalent. 
    """
    return torch.fft.fftshift(x)

def iftshift(x):
    """
    Places the DC component as the zero-component of the image. 
    """
    return torch.fft.ifftshift(x)

def flip(x):
    """
    Flip the input column-wise, to match the convention of CASA. 

    Args:
        x (torch.Tensor): 2D input

    Returns:
        torch.Tensor: Flipped version of x. 
    """
    return torch.flip(x, dims = [-1])

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

def noise_padding_dev(x, pad, sigma):
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


def zero_padding(x, pad):
    """
    Zero-pad a 2D input x using the padding given as argument. 

    Args:
        x (torch.Tensor): 2D image. 
        pad (Tuple): amount of padding in each direction around x. In PyTorch's convention, (pad_left, pad_right, pad_bot, pad_top) 

    Returns:
        torch.Tensor: zero-padded version of x. 
    """
    out = torch.nn.functional.pad(x, pad)
    return out

def noise_padding(x, pad, sigma):
    """
    Noise pad a 2D input x using the padding and the sigma given as argument. The sigma corresponds to the std of the perturbation's kernel 
    Gaussian noise in the padded region.

    Args:
        x (torch.Tensor): 2D input
        pad (Tuple): amount of padding in each direction around x. In PyTorch's convention, (pad_left, pad_right, pad_bot, pad_top) 
        sigma (function): Function computing the std of the perturbation kernel associated to the SDE used. Accessible through the trained score-based model. 

    Returns:
        torch.Tensor: noise-padded version of x. 
    """
    _, H, W = x.shape
    out = torch.nn.functional.pad(x, pad) 
    # Create a mask for padding region
    mask = torch.ones_like(out)
    pad_l, pad_r, pad_b, pad_t = pad
    mask[pad_t:pad_t + H, pad_l:pad_l+W] = 0.
    # Noise pad around the model
    z = torch.randn_like(out) * sigma
    out = out + z * mask
    return out



# def regular_padding(x, pad):
#     H, W = x.shape
#     out = torch.nn.functional.pad(x, (pad, pad, pad, pad))
#     return out

# def noise_padding(x, pad, sigma):
#     H, W = x.shape
#     out = torch.nn.functional.pad(x, (pad, pad, pad, pad)) 
#     # Create a mask for padding region
#     mask = torch.ones_like(out)
#     mask[pad:pad + H, pad:pad+W] = 0.
#     # Noise pad around the model
#     z = torch.randn_like(out) * sigma
#     out = out + z * mask
#     return out

def ancestral_model(t, x, score_model, model_parameters): 
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
    num_vis = sampling_function.sum()

    # To support batched input (note: the input is not batched during inference due to the vmap function)
    if len(x.shape) == 4: 
        N, _, H, W = x.shape
        sampling_function = sampling_function.tile(N, 1, 1)
    else: 
        N = 1
    x = link_function(x, B, C)   
    x_padded = noise_padding(x, pad = pad, sigma = sigma(t, score_model))
    vis_sampled = ft(x_padded).squeeze()[sampling_function] # some troublesome bug makes the squeeze needed here
    
    vis_sampled = vis_sampled.reshape(N, num_vis)
    y_hat = complex_to_real(vis_sampled)
    return y_hat

def old_model(t, x, score_model, model_parameters): 
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
  
    x_padded = noise_padding(x.squeeze(), pad = pad, sigma = sigma(t, score_model))
    x_padded = link_function(x_padded, B=B, C=C)
    vis_sampled = ft(ftshift(x_padded)).squeeze()[sampling_function] # some troublesome bug makes the squeeze needed here
    y_hat = complex_to_real(vis_sampled)
    return y_hat


# TODO: ADD OPTION FOR BATCHED INPUT (in order to forward model multiple  
def model(t, x, score_model, model_parameters): 
    """Apply the physical model associated to a simplified version of a radiotelescope's measurement process. 
    For stability reasons and to increase the resolution in Fourier space, noise padding was used. 

    Args:
        t (torch.Tensor): temperature in the sampling procedure of diffusion models.
        x (torch.Tensor): ground-truth 
        score_model (torch.Tensor): trained score-based model (= the score of a prior)
        model_parameters (Tuple): list of parameters for the model (sampling_function, B, C)
          - index 0: sampling function (mask selecting the measured visibilities in Fourier space, must have a shape (H, W) where H and W are the height
            and width of the padded image respectively)
          - index 1 and index 2: B and C, the link_function parameters (see function link_function)

    Returns:
        y_hat (torch.Tensor): model prediction
    """
        
    sampling_function, B, C, pad, padding_mode = model_parameters
    if padding_mode == "noise":
        padding = lambda x, pad: noise_padding(x, pad, sigma = sigma(t, score_model))
    elif padding_mode == "zero": 
        padding = zero_padding
    x = flip(x) # Enforcing good orientation for the final image
    # x_padded = noise_padding(x.squeeze(), pad = pad, sigma = sigma(t, score_model))
    # x_padded = regular_padding(x, pad = pad)
    x_padded = padding(x, pad)
    x_padded = link_function(x_padded, B=B, C=C) # score model units to physical units (#TODO find the right link function)
    vis_full = ft(iftshift(x_padded)) # iftshift to place the DC component at (0,0), as expected by torch.fft.fft2
    vis_sampled = ftshift(vis_full).squeeze()[sampling_function] # DC component at the center of the image then mask with the sampling function (whose DC component is at the center of the image)
    y_hat = complex_to_real(vis_sampled) # complex to vectorized real representation.
    return y_hat

def log_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters):
    """
    Calculate the log-likelihood of a gaussian distribution 
    Arguments: 
        y = processed gridded visibilities (real part and imaginary part concatenated)
        x = sky brightness 
        t = diffusion temperature
        A = linear model (sampling function and FT)  
    
    Returns: 
        log-likelihood of a gaussian distribution
    """ 
    y_hat = forward_model(t, x, score_model, model_parameters)
    Gamma_diag = torch.ones_like(y, device = y.device)/2 # Make sure to use the same notation as in the paper. 
    Gamma_diag[0] = 1 

    var = sigma(t, score_model) **2 * Gamma_diag + mu(t, score_model)**2 * sigma_y**2 ## sigma(t) ** 2 * AA^T + sigma_y ** 2
    res = (mu(t, score_model) * y - y_hat) ** 2 / var
    log_prob = -0.5 * torch.sum(res)
    return log_prob

def score_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters):
    return vmap(grad(lambda x, t: log_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters)), randomness = "different")(x, t)

# def model(t, x, score_model, model_parameters): 
#     """Apply the physical model associated to a simplified version of a radiotelescope's measurement process. 
#     For stability reasons and to increase the resolution in Fourier space, noise padding was used. 

#     Args:
#         t (torch.Tensor): temperature in the sampling procedure of diffusion models.
#         x (torch.Tensor): ground-truth 
#         score_model (torch.Tensor): trained score-based model (= the score of a prior)
#         model_parameters (Tuple): list of parameters for the model (sampling_function, B, C)
#           - index 0: sampling function (mask selecting the measured visibilities in Fourier space, must have a shape (H, W) where H and W are the height
#             and width of the padded image respectively)
#           - index 1 and index 2: B and C, the link_function parameters (see function link_function)

#     Returns:
#         y_hat (torch.Tensor): model prediction
#     """
        
#     sampling_function, B, C, pad = model_parameters
#     x = torch.flip(x, dims=[-1]) # Enforcing good orientation for the final image
#     x_padded = noise_padding(x.squeeze(), pad = pad, sigma = sigma(t, score_model))
#     x_padded = link_function(x_padded, B=B, C=C) # score model units to physical units (#TODO find the right link function)
#     vis_full = ft(iftshift(x_padded)) # iftshift to place the DC component at (0,0) as expected by torch.fft.fft2
#     vis_sampled = ftshift(vis_full).squeeze()[sampling_function] # DC component at the center of the image then mask with the sampling function
#     y_hat = complex_to_real(vis_sampled) # complex to vectorized real representation.
#     return y_hat

# def log_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters):
#     """
#     Calculate the log-likelihood of a gaussian distribution 
#     Arguments: 
#         y = processed gridded visibilities (real part and imaginary part concatenated)
#         x = sky brightness 
#         t = diffusion temperature
#         A = linear model (sampling function and FT)  
    
#     Returns: 
#         log-likelihood of a gaussian distribution
#     """ 
#     y_hat = forward_model(t, x, score_model, model_parameters)
#     Gamma_diag = torch.ones_like(y, device = y.device)/2
#     Gamma_diag[0] = 1 

#     var = sigma(t, score_model) **2 * Gamma_diag + mu(t, score_model)**2 * sigma_y**2 ## sigma(t) ** 2 * AA^T + sigma_y ** 2
#     res = (mu(t, score_model) * y - y_hat) ** 2 / var
#     log_prob = -0.5 * torch.sum(res)
#     return log_prob

# def score_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters):
#     return vmap(grad(lambda x, t: log_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters)), randomness = "different")(x, t)

def model_to_plot(t, x, score_model, model_parameters):
    """
    Same function as model(*args) except that the sampling function is replaced by an integer matrix where each pixel can either take
    the value 0 or 1. This function is only meant for visualization. 
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

def old_log_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters):
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
    y_hat = forward_model(t, x, score_model, model_parameters) 
    
    # Variance computed analytically with the convolved likelihood approximation
    var = sigma(t, score_model) ** 2/2 + mu(t, score_model)**2 * sigma_y ** 2
    var_0 = sigma(t, score_model) ** 2 + mu(t, score_model)**2 * sigma_y ** 2
    diff = (y_hat - mu(t, score_model) * y)**2 / var
    diff[0] = (y_hat[0] - mu(t, score_model) * y[0])**2 / var_0 
    
    # Log probability
    log_prob = - 0.5 * torch.sum(diff)
    return log_prob

def old_score_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters): 
    """
    Computes the score of the likelihood
    See log_likelihood(*args) for a description of the arguments 
    """ 
    return vmap(grad(lambda x, t: log_likelihood(t, x, y, sigma_y, forward_model, score_model, model_parameters)), randomness = "different")(x, t) # Do not interchange x and t

