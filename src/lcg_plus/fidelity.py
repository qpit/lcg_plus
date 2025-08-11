# Copyright © 2025 Technical University of Denmark

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from mpmath import mp
from scipy.special import logsumexp

def overlap_bosonic(state1, state2):
    #Equal number of modes check
    if state1.num_modes != state2.num_modes:
        raise ValueError('Number of modes is not the same in both states.')
    if state1.hbar != state2.hbar:
        raise ValueError("hbar is not the same in both states.")

    if state1.num_k != state1.num_weights and state2.num_k != state2.num_weights:
        return overlap_reduced(state1, state2)
    else:
        return overlap_full(state1, state2)

def overlap_log(state1, state2):
    #Equal number of modes check
    if state1.num_modes != state2.num_modes:
        raise ValueError('Number of modes is not the same in both states.')

    if state1.hbar != state2.hbar:
        raise ValueError("hbar is not the same in both states.")

    if state1.num_k != state1.num_weights and state2.num_k != state2.num_weights:
        return overlap_reduced_log(state1, state2)
    else:
        return overlap_full_log(state1, state2)
        

def overlap_full(state1, state2):
    """Calculate the overlap of two states in sum-of-gaussian representation.
    If state1 == state2, returns the purity.
    
    Args:
        state1
        state2
    Returns:
        overlap (complex): fidelity of state1 with state2, if one of them is pure
    """
    
    #Equal number of modes check
    if state1.num_modes != state2.num_modes:
        raise ValueError('Number of modes is not the same in both states.')

    N = state1.num_modes

    weights1 = state1.weights
    weights2 = state2.weights
    
    deltas = state1.means[:,np.newaxis,:] - state2.means[np.newaxis,:,:]

    if state1.num_covs == 1 and state2.num_covs == 1:
        covsum = state1.covs + state2.covs #covs are the same shape, so no broadcasting needed
    else:
        covsum = state1.covs[:,np.newaxis,:] + state2.covs[np.newaxis,:,:] 
        
    covsum_inv = np.linalg.inv(covsum)
    exp_arg = np.einsum('...j,...jk,...k', deltas, covsum_inv, deltas)
  
    weighted_exp = ( state1.weights[:,np.newaxis] * state2.weights[np.newaxis,:] * state1.hbar ** N
               * np.exp( -0.5 * exp_arg) / np.sqrt( np.linalg.det(covsum)) )
               
    overlap = np.sum(weighted_exp)/(state1.norm*state2.norm)
    
    return overlap

def overlap_full_log(state1, state2):
    """Calculate the overlap of two states in sum-of-gaussian representation.
    If state1 == state2, returns the purity.
    
    Args:
        state1
        state2
    Returns:
        overlap (complex): fidelity of state1 with state2, if one of them is pure
    """
    
    #Equal number of modes check
    if state1.num_modes != state2.num_modes:
        raise ValueError('Number of modes is not the same in both states.')

    N = state1.num_modes

    #weights1 = state1.weights
    #weights2 = state2.weights

    lw1 = state1.log_weights
    lw2 = state2.log_weights

    n1 = state1.norm
    n2 = state2.norm

    deltas = state1.means[:,np.newaxis,:] - state2.means[np.newaxis,:,:]

    if state1.num_covs == 1 and state2.num_covs == 1:
        covsum = state1.covs + state2.covs #covs are the same shape, so no broadcasting needed
    else:
        covsum = state1.covs[:,np.newaxis,:] + state2.covs[np.newaxis,:,:] 

    covsum_inv = np.linalg.inv(covsum)
    exp_arg = np.einsum('...j,...jk,...k', deltas, covsum_inv, deltas)

    weighted_exp = lw1[:,np.newaxis] + lw2[np.newaxis,:] - 0.5 * exp_arg - 0.5 * np.log(np.linalg.det(covsum))
  
    #weighted_exp = ( state1.weights[:,np.newaxis] * state2.weights[np.newaxis,:] * hbar ** N
               #* np.exp( -0.5 * exp_arg) / np.sqrt( np.linalg.det(covsum)) )
    overlap = np.exp(logsumexp(weighted_exp))/(n1*n2) * state1.hbar ** N
               
    #overlap = np.sum(weighted_exp)/(state1.norm*state2.norm)
    
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

    weighted_exp = weights1[:,np.newaxis] * weights2[np.newaxis,:]*state1.hbar ** N* np.exp( -0.5 * exp_arg) / np.sqrt( np.linalg.det(covsum)) 

    weighted_exp_special = weights1[k1:,np.newaxis] * np.conjugate(
        weights2[np.newaxis,k2:]) *state1.hbar**N *np.exp(-0.5*exp_arg_special)/np.sqrt( np.linalg.det(covsum))  
                
    
    overlap = 0    
    overlap += np.sum(weighted_exp[0:k1,0:k2])
    overlap += np.sum(weighted_exp[0:k1,k2:]).real
    overlap += np.sum(weighted_exp[k1:,0:k2]).real
    overlap += 0.5*np.sum(weighted_exp[k1:,k2:]).real
    overlap += 0.5*np.sum(weighted_exp_special).real
    
    return overlap/(state1.norm*state2.norm) 

def overlap_reduced_log(state1, state2):
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

    #weights1 = state1.weights
    #weights2 = state2.weights
    lw1 = state1.log_weights
    lw2 = state2.log_weights
    
    deltas = state1.means[:,np.newaxis,:] - state2.means[np.newaxis,:,:]
    deltas_special = state1.means[k1:,np.newaxis,:] - np.conjugate(state2.means[np.newaxis,k2:,:])

    covsum = state1.covs + state2.covs #covs are the same shape, so no broadcasting needed
    
    covsum_inv = np.linalg.inv(covsum)
    
    exp_arg = -0.5 * np.einsum('...j,...jk,...k', deltas, covsum_inv, deltas)
    exp_arg_special = -0.5 * np.einsum('...j,...jk,...k', deltas_special, covsum_inv, deltas_special)
    
    prefactor = -0.5*np.log(np.linalg.det(covsum))
    
    weighted_exp = lw1[:,np.newaxis] + lw2[np.newaxis,:] + exp_arg + prefactor
    weighted_exp_special = lw1[k1:,np.newaxis] + np.conjugate(lw2[np.newaxis,k2:]) + exp_arg_special + prefactor

    #weighted_exp = weights1[:,np.newaxis] * weights2[np.newaxis,:]*hbar ** N* np.exp( -0.5 * exp_arg) / np.sqrt( np.linalg.det(covsum)) 

    #weighted_exp_special = weights1[k1:,np.newaxis] * np.conjugate(
        #weights2[np.newaxis,k2:]) *hbar**N *np.exp(-0.5*exp_arg_special)/np.sqrt( np.linalg.det(covsum))  
                
    
    overlap = 0    
    
    overlap += np.exp(logsumexp(weighted_exp[0:k1,0:k2]))
    overlap += np.exp(logsumexp(weighted_exp[0:k1,k2:])).real
    overlap += np.exp(logsumexp(weighted_exp[k1:,0:k2])).real
    overlap += 0.5*np.exp(logsumexp(weighted_exp[k1:,k2:])).real
    overlap += 0.5*np.exp(logsumexp(weighted_exp_special)).real
    
    
    #overlap += np.sum(weighted_exp[0:k1,0:k2])
    #overlap += np.sum(weighted_exp[0:k1,k2:]).real
    #overlap += np.sum(weighted_exp[k1:,0:k2]).real
    #overlap += 0.5*np.sum(weighted_exp[k1:,k2:]).real
    #overlap += 0.5*np.sum(weighted_exp_special).real
    
    return state1.hbar**N*overlap/(state1.norm*state2.norm)



def overlap_with_wigner(W1, W2, xvec, pvec, hbar = 2):
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