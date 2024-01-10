from score_models import ScoreModel
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from torch.func import vmap, grad 
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"


def complex_to_real(z):
    """Convert a complex tensor into a real representation where the real
    part and the imaginary part are concatenated 

    Args:
        z (torch.Tensor): complex tensor of dimension D 

    Returns:
        real tensor of dimensions 2*D
    """
    return torch.cat([z.real, z.imag])   

# def real_to_complex(x):
#     """Convert a real representation of a complex tensor into its associated 
#     complex tensor (see the description of the complex_to_real(*args) function)

#     Args:
#         z (torch.Tensor): real tensor of dimension 2*D 

#     Returns:
#         complex tensor of dimensions D
#     """
#     H, W = x.shape
#     D = H//2 # width of the image in its complex representation
#     return x[:D] + 1j * x[D:]

def real_to_complex(x):
    """Convert a real representation of a complex tensor into its associated 
    complex tensor (see the description of the complex_to_real(*args) function)

    Args:
        z (torch.Tensor): real tensor of dimension 2*D 

    Returns:
        complex tensor of dimensions D
    """
    C, H, W = x.shape
    D = H//2 # width of the image in its complex representation
    return x[0] + 1j * x[1]

# score_prior --> score-based model, score_likelihood --> convolved likelihood approximation (see appendix https://arxiv.org/abs/2311.18012)
def sigma(t, score_prior): 
    return score_prior.sde.sigma(t)

def mu(t, score_prior): 
    return score_prior.sde.marginal_prob_scalars(t)[0]

def drift_fn(t, x, score_prior): 
    return score_prior.sde.drift(t, x)

def g(t, x, score_prior): 
    return score_prior.sde.diffusion(t, x)

def identity_model(x, t): 
    return x

def score_posterior(t, x, y, sigma_y, forward_model, score_model, score_likelihood, model_parameters, tweedie):
    if tweedie: 
        sigma_t = sigma(t, score_model)[0].item() # During sampling every sample is evaluated at the same temperature so there's no issue with this
        mu_t = mu(t, score_model)[0].item()
        #tweedie_x = (x + sigma(t, score_model)**2 * score_model.score(t, x))/ mu(t, score_model)
        tweedie_x = (x + sigma_t ** 2 * score_model.score(t, x)) / mu_t
    else: 
        tweedie_x = x
    score_prior = score_model.score(t, x)
    score_lh = score_likelihood(t, tweedie_x, y, sigma_y, forward_model=forward_model, score_model=score_model, model_parameters = model_parameters)
    return score_prior + score_lh

def euler_sampler(y, sigma_y, forward_model, score_model, score_likelihood, model_parameters, num_samples, num_steps,  tweedie = False, keep_chain=False, test_time=False, img_size = (64, 64)):
    """
    Discretization of the Euler-Maruyama sampler 

    Args:
        y (array): Observation
        sigma_y (float): std of an isotropic gaussian that we sample to perturb the observation y
        forward_model (function): physical model mapping a ground-truth x to a model \hat{y} 
        score_model (function): Trained score-based model playing the role of a prior 
        score_likelihood (function): see function score_likelihood 
        model_parameters (tuple): parameters of the function score_likelihood
        num_samples (int): number of samples to generate
        num_steps (int): number of steps during the sampling procedure
        tweedie (bool, optional): To enable a correction of the score of the posterior with Tweedie's formula. Defaults to False.
        keep_chain (bool, optional): To analyze possible anomalies that may occur during the sampling procedure. Defaults to False.
        test_time (bool, optional): Runs the loop for 20 iterations to evaluate time required for more iterations. Defaults to False.
        img_size (tuple, optional): image size of the ground-truth x. Defaults to (64, 64) (= simulation)

    Returns:
        array: posterior samples 
    """
    t = torch.ones(size = (num_samples,1)).to(device)
    sigma_max = sigma(t, score_model)[0]
    x = sigma_max * torch.randn([num_samples, 1, *img_size]).to(device)
    dt = -1/num_steps 
    
    chain = []
    with torch.no_grad(): 
        for i in (pbar := tqdm(range(num_steps - 1))):
            pbar.set_description(f"t = {t[0].item():.2f} | scale ~ {x.std():.2e} | sigma(t) = {sigma(t, score_model)[0].item():.2e} | mu(t) = {mu(t, score_model)[0].item():.2e}")
            z = torch.randn_like(x).to(device)
            gradient =  score_posterior(t, x, y, sigma_y, forward_model, score_model, score_likelihood, model_parameters, tweedie = tweedie)
            drift = drift_fn(t, x, score_model)
            diffusion = g(t, x, score_model)
            x_mean  = x + drift * dt - diffusion ** 2 * gradient * dt
            noise = diffusion * (-dt) ** 0.5 * z
            x = x_mean + noise
            t += dt

            if torch.isnan(x).any().item(): 
                print("Nans appearing")
                break
            if keep_chain: 
                chain.append(x.cpu().numpy())

            if test_time: 
                if i==20: 
                    break

    if keep_chain: 
        return x_mean, chain
     
    else: 
        return x_mean 

def pc_sampler(y, sigma_y, num_samples, num_pred_steps, num_corr_steps, score_function, snr = 1e-2, img_size = 28): 
        t = torch.ones(size = (num_samples, 1)).to(device)
        x = sigma(t, score_prior) * torch.randn([num_samples, img_size ** 2]).to(device)
        dt = -1/num_pred_steps

        with torch.no_grad(): 
            for _ in tqdm(range(num_pred_steps-1)): 
                # Corrector step: (Only if we are not at 0 temperature )
                gradient = score_function(y, x, t, sigma_y)
                for _ in range(num_corr_steps): 
                    z = torch.randn_like(x)
                    grad_norm = torch.mean(torch.norm(gradient, dim = -1)) # mean of the norm of the score over the batch 
                    noise_norm = torch.mean(torch.norm(z, dim = -1))
                    epsilon =  2 * (snr * noise_norm / grad_norm) ** 2
                    x = x + epsilon * gradient + (2 * epsilon) ** 0.5 * z * dt  

                # Predictor step: 
                z = torch.randn_like(x).to(device)
                gradient = score_function(y, x, t, sigma_y)
                drift = drift_fn(t, x, score_prior)
                diffusion = g(t, x, score_prior)
                x_mean = x + drift * dt - diffusion**2 * gradient * dt  
                noise = diffusion * (-dt) ** 0.5 * z
                x = x_mean + noise
                t += dt

                if torch.isnan(x).any().item(): 
                    print("Nans appearing, stopping sampling...")
                    break
                
        return x_mean          



