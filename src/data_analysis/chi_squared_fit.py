import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize

def chi2_pdf(x, k): 
    return chi2.pdf(x, k)

def neg_log_likelihood(k, x): 
    """
    Computes the negative log-likelihood of the chi squared distribution for a 
    certain number of degrees of freedom k given some observed samples x.  

    Args:
        k (float): Number of degrees of freedom of the chi squared distribution
        x (np.array): Observed samples

    Returns:
        np.array: negative log-likelihood for the given parameters. 
    """
    return -np.sum(np.log(chi2_pdf(x, k)))

def fit_chi2(k0, x): 
    """
    Fits a chi-squared distribution on the observed samples x

    Args:
        k0 (float): Initial guess for the number of degrees of freedom of the underlying chi-squared distribution (parameter to fit)
        x (np.array): Observed samples

    Returns:
        Output of the minimize function of scipy.stats (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)
    """
    result = minimize(neg_log_likelihood, x0 = k0, args = (x,))
    return result