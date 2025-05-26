import numpy as np
from mpmath import mp
from copy import copy
from scipy.special import logsumexp
def get_upbnd_weights(means_quad, covs_quad, log_weights, method = 'normal'):
    """Upper bound distribution according to method
    """
    # Indices of the Gaussians in the linear combination with imaginary means
    imag_means_ind = np.where(means_quad.imag.any(axis=1))[0]

    #Indices of Gaussians in the linear combination with real means
    real_means_ind = np.where(means_quad.imag == 0)[0]
    
    nonneg_weights_ind = np.where(np.array(log_weights.imag // np.pi == 0))[0]
    
    # Union of the two sets forms the set of indices used to construct the
    # upper bounding function
    
    if method == 'normal':
        ub_ind = np.union1d(imag_means_ind, nonneg_weights_ind)
    elif method == 'coherent':
        ub_ind = real_means_ind
    elif method == 'ignore_imag_prefactor':
        ub_ind = np.union1d(imag_means_ind, nonneg_weights_ind)
    elif method == 'imaginary':
        ub_ind = imag_means_ind

    # Build weights for the upper bounding function
    
    # Take absolute value of all weights (real part of exponentials)
    
    ub_weights = copy(log_weights.real)
    
    # If there are terms with complex means, multiply the associated weights
    # by an extra prefactor, which comes from the cross term between the
    # imaginary parts of the means
    if method == 'normal':
        imag_means = means_quad.imag
        # Construct prefactor
        imag_exp_arg = 0.5 * np.einsum(
            "...j,...jk,...k",
            imag_means,
            np.linalg.inv(covs_quad),
            imag_means,
        )
        
        ub_weights += imag_exp_arg
            
    # Keep only the weights that are indexed by ub_ind
    ub_weights = ub_weights[ub_ind]
    
    # To define a probability dsitribution, normalize the set of weights
    
    ub_weights_prob = np.exp(ub_weights - logsumexp(ub_weights))
    
    return ub_ind, ub_weights, ub_weights_prob

def generaldyne_probability(sample, means_quad, covs_quad, log_weights, prec = False):
    """Evaluate probability distribution at sample, given means, covs and weights. 
    """
 
    diff_sample = sample - means_quad
    
    # Calculate arguments for the Gaussian functions used to calculate
    # the exact probability distribution at the sampled point
    exp_arg = np.einsum(
        "...j,...jk,...k",
        diff_sample,
        np.linalg.inv(covs_quad),
        diff_sample,
    )

    prefactors = 1 / np.sqrt(2 * np.pi * np.linalg.det(covs_quad))
    
    log = log_weights - 0.5 * exp_arg

    if prec:
        #Calculate sum of exponentials with high precision using mpmath
        prob_dist_val = prefactors * float( mp.re(mp.fsum([mp.exp(i) for i in log]))) 
        
    else:
        
        prob_dist_val = prefactors * np.exp(logsumexp(log))
    
    
    #prob_dist_val = np.sum(weights * prefactors * np.exp(-0.5 * exp_arg))

    prob_dist_val = np.real_if_close(prob_dist_val)
    
    return prob_dist_val #May be complex in fast rep

def select_quads(self, modes, covmat =[]):
    """Make the quadrature selection for measurement on some modes. If covmat == [], do x-measurement.
    """
    #Indices for the relevant quadratures
    quad_ind = 2 * np.array(modes) #x quadrature
    if covmat: 
        quad_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1)) #Both quadratures
    
    # Associated means and covs
    means_quad = self.means[:, quad_ind]
    covs_quad = self.covs[:, quad_ind, :][:, :, quad_ind].real
    
    if covmat:
        covs_quad += covmat #Add generaldyne covmat if needed
    return means_quad, covs_quad, quad_ind

def get_upbnd_gaussian(self, means_quad, covs_quad, quad_ind, prec =False):
    """Get upper bound Gaussian distribution from first and second moments.
    """

    #Calculate first and second moments
    mu = self.get_mean(MP)
    sigma = self.get_cov(MP)

    #Select the proper quad indices
    mu = np.array(mu[quad_ind])
    sigma = np.array([sigma[quad_ind,quad_ind]])

    #Calculate tentatitive scaling factor (be careful: the distribution could have an envelope that is not a guassian,
    # or might be zero at the center (e.g. p-quadrature of a cat with parity 1)).
    scale = generaldyne_probability(mu, means_quad, covs_quad, self.log_weights, prec).real
    
    return sigma, mu, scale

