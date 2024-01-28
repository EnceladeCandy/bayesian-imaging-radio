import torch 
import numpy as np 


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