import numpy as np
from mpmath import mp

hbar = 2

def overlap_bosonic(state1, state2):
    #Equal number of modes check
    if state1.num_modes != state2.num_modes:
        raise ValueError('Number of modes is not the same in both states.')

    if state1.num_k != state1.num_weights or state2.num_k != state2.num_weights:
        return overlap_reduced(state1, state2)
    else:
        return overlap_full(state1, state2)
        

def overlap_full(state1, state2, MP = False):
    """Calculate the overlap of two states in sum-of-gaussian representation.
    If state1 == state2, returns the purity.
    
    Args:
        state1
        state2
        MP (bool): Higher precision. Not completely implemented yet. 
    Returns:
        overlap (complex): fidelity of state1 with state2, assuming one of them is pure
    """
    
    #Equal number of modes check
    if state1.num_modes != state2.num_modes:
        raise ValueError('Number of modes is not the same in both states.')

    N = state1.num_modes

    weights1 = state1.weights
    weights2 = state2.weights
    
    if MP:
        norm1 = mp.fsum(weights1)
        norm2 = mp.fsum(weights2)

    deltas = state1.means[:,np.newaxis,:] - state2.means[np.newaxis,:,:]

    if state1.num_covs == 1 or state2.num_covs == 1:
        covsum = state1.covs + state2.covs #covs are the same shape, so no broadcasting needed
    else:
        covsum = state1.covs[:,np.newaxis,:] + state2.covs[np.newaxis,:,:] 

    covsum_inv = np.linalg.inv(covsum)
    exp_arg = np.einsum('...j,...jk,...k', deltas, covsum_inv, deltas)
  
    weighted_exp = ( state1.weights[:,np.newaxis] * state2.weights[np.newaxis,:] * hbar ** N
               * np.exp( -0.5 * exp_arg) / np.sqrt( np.linalg.det(covsum)) )
               
    overlap = np.sum(weighted_exp)/(state1.norm*state2.norm)
    
    return overlap
    
def overlap_reduced(state1, state2):
    """Overlap of two states when one or more are in the redcued format,
    where only half the cross terms are stored. 

    If one of the states is pure, then this is the fidelity.
    
    There is only one covariance matrix, but many weights and means.
    
    Args:
        state1 (bosonicplus.base.State):
        state2 (bosonicplus.base.State):
        
    Returns:
        overlap (complex): overlap of state1 with state2
    """
    
    #Equal number of modes check
    if state1.num_modes != state2.num_modes:
        raise ValueError('Number of modes is not the same in both states.')

    N = state1.num_modes

    
    #The number of weights that have real Gaussians
    k1 = state1.num_k 
    k2 = state2.num_k 

    weights1 = state1.weights
    weights2 = state2.weights
    
    deltas = state1.means[:,np.newaxis,:] - state2.means[np.newaxis,:,:]
    deltas_special = state1.means[k1:,np.newaxis,:] - np.conjugate(state2.means[np.newaxis,k2:,:])

    covsum = state1.covs + state2.covs #covs are the same shape, so no broadcasting needed
    
    covsum_inv = np.linalg.inv(covsum)
    
    exp_arg = np.einsum('...j,...jk,...k', deltas, covsum_inv, deltas)
    exp_arg_special = np.einsum('...j,...jk,...k', deltas_special, covsum_inv, deltas_special)

    weighted_exp = weights1[:,np.newaxis] * weights2[np.newaxis,:]*hbar ** N* np.exp( -0.5 * exp_arg) / np.sqrt( np.linalg.det(covsum)) 

    weighted_exp_special = weights1[k1:,np.newaxis] * np.conjugate(
        weights2[np.newaxis,k2:]) *hbar**N *np.exp(-0.5*exp_arg_special)/np.sqrt( np.linalg.det(covsum))  
                
    
    overlap = 0    
    overlap += np.sum(weighted_exp[0:k1,0:k2])
    overlap += np.sum(weighted_exp[0:k1,k2:]).real
    overlap += np.sum(weighted_exp[k1:,0:k2]).real
    overlap += 0.5*np.sum(weighted_exp[k1:,k2:]).real
    overlap += 0.5*np.sum(weighted_exp_special).real
    
    return overlap/(state1.norm*state2.norm)



def overlap_with_wigner(W1, W2, xvec, pvec):
    """
    Calculate overlap by numerically integrating over the explicit single-mode Wigner functions.
    
    Args:
        W1 (ndarray): Wigner mesh of state 1
        W2 (ndarray): Wigner mesh of state 2
        xvec (ndarray): points in x
        pvec (ndarray): points in p
        
    Returns:
        (float): overlap
    """
    delta_x = xvec[-1] - xvec[-2]
    delta_p = pvec[-1] - pvec[-2]
    
    return 2 * np.pi * hbar * np.sum(W1*W2*delta_x*delta_p)