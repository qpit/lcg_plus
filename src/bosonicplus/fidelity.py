import numpy as np

hbar = 2

def fidelity_coherent(state1, state2):
    """Calculate fidelity between two states in coherent representation 
    i.e. there is only one covariance matrix, but many weights and means.
    
    Args:
        
    Returns:
        fidelity (complex): fidelity of state1 with state2, assuming one of them is pure
    """
    
    #Equal number of modes check
    if state1.num_modes != state2.num_modes:
        raise ValueError('Number of modes is not the same in both states.')

    N = state1.num_modes

    weights1 = state1.weights
    weights2 = state2.weights
    
    deltas = state1.means[:,np.newaxis,:] - state2.means[np.newaxis,:,:]

    if state1.num_covs == 1 or state2.num_covs == 1:
        covsum = state1.covs + state2.covs #covs are the same shape, so no broadcasting needed
    else:
        covsum = state1.covs[:,np.newaxis,:] + state2.covs[np.newaxis,:,:] 

    covsum_inv = np.linalg.inv(covsum)
    
    exp_arg = np.einsum('...j,...jk,...k', deltas, covsum_inv, deltas)
  
    weighted_exp = ( state1.weights[:,np.newaxis] * state2.weights[np.newaxis,:] * hbar ** N
               * np.exp( -0.5 * exp_arg) / np.sqrt( np.linalg.det(covsum)) )
               
    fidelity = np.sum(weighted_exp)
    
    return fidelity



def fidelity_with_wigner(W1, W2, xvec, pvec):
    """
    Calculate fidelity via numerically integrating over explicit Wigner functions.
    
    Args:
        W1 (ndarray): Wigner mesh
        W2 (ndarray):
        xvec (ndarray): points in x
        pvec (ndarray): points in p
        
    Returns:
        (float): fidelity
    """
    delta_x = xvec[-1] - xvec[-2]
    delta_p = pvec[-1] - pvec[-2]
    
    return 2 * np.pi * hbar * np.sum(W1*W2*delta_x*delta_p)