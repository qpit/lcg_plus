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
from scipy.special import comb
from lcg_plus.states.coherent import gen_fock_coherent, gen_fock_bosonic, gen_fock_coherent_old, gen_fock_log
from lcg_plus.from_sf import chop_in_blocks_multi, chop_in_blocks_vector_multi, chop_in_blocks_multi_v2, chop_in_blocks_vector_multi_v2
from scipy.special import logsumexp


# PNRD MEASUREMENT
# ------------------------------------
def project_fock_coherent(n, data, mode, inf=1e-4, k2=None, hbar =2 ):  
    """Returns data tuple after projecting mode on fock state n (in coherent approx)
    
    Args:
        n (int): photon number
        data (tuple) : [means, covs, weights]
        mode (int): mode index that is measured with PNRD
        inf (float): infidelity of the fock approx
        k1 (int): number of "normal" gaussians in fock POVM
        k2 (int): number of "normal" gaussians in state
        
    Returns:
        data_A (tuple): 
        prob (float): probability of the measurement
    """
    means, covs, log_weights = data
    if k2: 
        means_f, sigma_f, log_weights_f, k1 = gen_fock_coherent(n, inf, fast=True)
    else:
        #means_f, sigma_f, log_weights_f, k1 = gen_fock_coherent_old(n, inf)
        means_f, sigma_f, log_weights_f, k1 = gen_fock_coherent(n, inf, fast=False)
    modes = [mode]

    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))
    
    sigma_A, sigma_AB, sigma_B = chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = chop_in_blocks_vector_multi(means, mode_ind)
    
    #New array sizes
    M = len(means_f)
    N = len(r_A)
    L = len(r_A[0,:])
    
    C = sigma_B + sigma_f
    C_inv = np.linalg.inv(C)

    sigma_ABC = sigma_AB @ C_inv 
    
    sigma_A_tilde = sigma_ABC @ sigma_AB.transpose(0, 2, 1)
    sigma_A_prime = sigma_A - sigma_A_tilde

    delta_B = means_f[np.newaxis,:,:] - r_B[:,np.newaxis,:]
    r_A_prime = r_A[:,np.newaxis,:] + np.einsum("...jk,...k", sigma_ABC, delta_B) 

    reweights_exp_arg = -0.5 * np.einsum("...j,...jk,...k", delta_B, C_inv[np.newaxis,:,:], delta_B)
    #print('Shape reweights_exp_arg' ,reweights_exp_arg.shape)

    #Norm factor
    prefactor = np.log(2*np.pi*hbar / np.sqrt(np.linalg.det(2*np.pi* C)))
    #Norm = (2*np.pi*hbar)/ (np.sqrt(np.linalg.det( 2* np.pi * C))) 
    
    #reweights = log_weights_f[np.newaxis,:] + log_weights[:,np.newaxis] - Norm + reweights_exp_arg
    reweights = log_weights_f[np.newaxis,:] + log_weights[:,np.newaxis] + reweights_exp_arg + prefactor
    #reweights = log_weights_f[np.newaxis,:] + log_weights[:,np.newaxis] + reweights_exp_arg 


    if k2 and k2>1:
        
        delta_B_special = means_f[np.newaxis,k1:] - np.conjugate(r_B[k2:,np.newaxis,:])
        r_A_prime_special = np.conjugate(r_A[k2:,np.newaxis,:]) + np.einsum("...jk,...k", sigma_ABC, delta_B_special)
        reweights_exp_arg_special = -0.5*np.einsum("...j,...jk,...k", delta_B_special, C_inv[np.newaxis,:,:], delta_B_special)
        
        #reweights_special = log_weights_f[np.newaxis,k1:]+np.conjugate(log_weights[k2:,np.newaxis]) - Norm + re_weights_exp_arg_special
        reweights_special = log_weights_f[np.newaxis,k1:]+np.conjugate(log_weights[k2:,np.newaxis]) + reweights_exp_arg_special + prefactor
        #reweights_special = log_weights_f[np.newaxis,k1:]+np.conjugate(log_weights[k2:,np.newaxis]) + reweights_exp_arg_special

        #Building the new data tuple with the k1*k2 normal gaussians appearing first
        
        w1 = reweights[0:k2,0:k1].flatten()
        w2 = reweights[0:k2,k1:].flatten()
        w3 = reweights[k2:,0:k1].flatten()
        w4 = reweights[k2:,k1:].flatten()-np.log(2)
        w5 = reweights_special.flatten()-np.log(2)
        
        reweights = np.concatenate([w1,w2,w3,w4,w5])

        m1 = r_A_prime[0:k2,0:k1,:]
        m2 = r_A_prime[0:k2,k1:,:]
        m3 = r_A_prime[k2:,0:k1,:]
        m4 = r_A_prime[k2:,k1:,:]
        m5 = r_A_prime_special
        
        m1=m1.reshape(-1,m1.shape[-1])
        m2=m2.reshape(-1,m2.shape[-1])
        m3=m3.reshape(-1,m3.shape[-1])
        m4=m4.reshape(-1,m4.shape[-1])
        m5=m5.reshape(-1,m5.shape[-1])
        
        r_A_prime = np.concatenate([m1,m2,m3,m4,m5],axis=0)
        
    else:
        reweights = reweights.reshape([M*N])
        r_A_prime = r_A_prime.reshape([M*N,L])
    if k2:  
        #prob = np.sum(np.exp(reweights).real)
        data_A = r_A_prime, sigma_A_prime, reweights, k1*k2
        return data_A
    else:
        #prob=np.sum(np.exp(reweights))
        data_A = r_A_prime, sigma_A_prime, reweights, len(reweights)
        
        return data_A


# PSEUDO PNRD MEASUREMENT
# --------------------------------

def pkn(k,n, N):
    """ n photon number probabilities of pseudo POVM with N on/off detectors and k clicks

    (To do: replace with Stirling number stuff)
    """
    ls = np.arange(k+1)
    pkn =  comb(N,k)*1/N**n * np.sum(comb(k,ls)*(-1)**ls * (k-ls)**n)
    return pkn


def ppnrd_povm_thermal(k, N, hbar =2):
    """Pseudo pnrd povm as weighted sum of thermal states. 
    
    Args: 
        k (int): Number of clicks
        N (int): Number of on/off detectors in pseudo PNRD

    Returns:
        data (tuple): data tuple of POVM (sum of Gaussians)
    """

    ls = np.arange(k+1)
    eta = (N - k + ls)/N
    if k == N:
        eta[0] = 1e-20 #close to zero to avoid inf
    
    nbars = (1 - eta)/eta
    log_weights = np.log(comb(N,k)) + np.log(comb(k,ls)) - np.log(eta) + 1j*np.pi*(ls%2)
    
    
    #weights = comb(N,k) *comb(k,ls)* (-1)**ls * 1/eta 
    
    covs = np.array([np.eye(2)*hbar/2*(1+2*nbar) for nbar in nbars])
    means = np.repeat( np.zeros(2)[np.newaxis,:], k+1, axis = 0)

    data = means, covs, log_weights
    
    return data

def project_ppnrd_thermal(data, mode, n, M, hbar = 2):
    
    means, covs, log_weights = data
    
    modes = [mode]
    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))

    sigma_A, sigma_AB, sigma_B = chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = chop_in_blocks_vector_multi(means, mode_ind)

    # pPNRD PNRD POVM with thermal states
    mu_povm, sigma_povm, log_weights_povm = ppnrd_povm_thermal(n, M)    

    #print(mu_povm.shape)
    #print(r_A.shape)

    # New data sizes
    M = len(mu_povm)
    N = len(sigma_A)
    L = len(r_A[0,:])

    # Use broadcasting to add arrays of different sizes
    C = (sigma_B[:,np.newaxis,:,:]+sigma_povm[np.newaxis,:,:,:])
    C_inv = np.linalg.inv(C)

    r_A_tilde = np.einsum("...jk,...kl,...l", sigma_AB[:,np.newaxis,:,:], C_inv, r_B[:,np.newaxis,:]) #OBS: mu_povm are zero
    #print(r_A_tilde.shape)
    r_A_prime = (r_A[:,np.newaxis,:] - r_A_tilde )
    r_A_prime = r_A_prime.reshape([M*N,L])
    
    sigma_A_tilde = np.einsum("...jk,...kl,...lm",sigma_AB[:,np.newaxis,:,:], C_inv, sigma_AB.transpose(0,2,1)[:,np.newaxis,:,:] )
    sigma_A_prime = (sigma_A[:,np.newaxis,:,:] - sigma_A_tilde)
    sigma_A_prime = sigma_A_prime.reshape([M*N,L,L])

    reweights_exp_arg = -0.5 * np.einsum("...j,...jk,...k", -r_B[:,np.newaxis,:], C_inv, -r_B[:,np.newaxis,:]) #OBS: mu_povm are zero
    prefactor = np.log( 2*np.pi*hbar/np.sqrt(np.linalg.det(2*np.pi*C)))
    #Norm = (2*np.pi*hbar) / (np.sqrt(np.linalg.det( 2* np.pi * C))) 
     
    #new_weights = log_weights_povm[np.newaxis,:] + log_weights[:,np.newaxis] - Norm + reweights_exp_arg
    new_weights = log_weights_povm[np.newaxis,:] + log_weights[:,np.newaxis] + reweights_exp_arg + prefactor

    new_weights = new_weights.reshape([M*N])

    #prob = np.sum(np.exp(new_weights))
    #new_weights /=  prob

    data_A = r_A_prime, sigma_A_prime, new_weights, len(new_weights)
    
    return data_A 

def project_fock_thermal(data, mode, n ,r = 0.05, hbar = 2):
    means, covs, log_weights = data
    
    modes = [mode]
    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))

    sigma_A, sigma_AB, sigma_B = chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = chop_in_blocks_vector_multi(means, mode_ind)

    # Fock state with thermal states
    mu_povm, sigma_povm, log_weights_povm = gen_fock_log(n, r)
   
    # New data sizes
    M = len(mu_povm)
    N = len(sigma_A)
    L = len(r_A[0,:])

    # Use broadcasting to add arrays of different sizes
    C = (sigma_B[:,np.newaxis,:,:]+sigma_povm[np.newaxis,:,:,:])
    C_inv = np.linalg.inv(C)

    r_A_tilde = np.einsum("...jk,...kl,...l", sigma_AB[:,np.newaxis,:,:], C_inv, r_B[:,np.newaxis,:]) #OBS: mu_povm are zero
    #print(r_A_tilde.shape)
    r_A_prime = (r_A[:,np.newaxis,:] - r_A_tilde )
    r_A_prime = r_A_prime.reshape([M*N,L])
    
    sigma_A_tilde = np.einsum("...jk,...kl,...lm",sigma_AB[:,np.newaxis,:,:], C_inv, sigma_AB.transpose(0,2,1)[:,np.newaxis,:,:] )
    sigma_A_prime = (sigma_A[:,np.newaxis,:,:] - sigma_A_tilde)
    sigma_A_prime = sigma_A_prime.reshape([M*N,L,L])

    reweights_exp_arg = -0.5 * np.einsum("...j,...jk,...k", -r_B[:,np.newaxis,:], C_inv, -r_B[:,np.newaxis,:]) #OBS: mu_povm are zero
    prefactor = np.log(2*np.pi*hbar/np.sqrt(np.linalg.det(2*np.pi*C)))
    #Norm = (2*np.pi*hbar) / (np.sqrt(np.linalg.det( 2* np.pi * C))) 
    #new_weights = weights_povm[np.newaxis,:] + weights[:,np.newaxis] - Norm + reweights_exp_arg
    
    new_weights = log_weights_povm[np.newaxis,:] + log_weights[:,np.newaxis] + reweights_exp_arg + prefactor
    new_weights = new_weights.reshape([M*N])

    #prob = np.sum(np.exp(new_weights))
    #new_weights /=  prob

    data_A = r_A_prime, sigma_A_prime, new_weights, len(new_weights)
    return data_A


# Homodyne measurement
# ----------------------------
def project_homodyne(data, mode, result, k):
    r"""Following Brask's Gaussian note (single mode)
    Do a homodyne x-measurement on one mode
    """

    means, covs, log_weights = data
    
    #For the position eigenstate POVM
    P = np.array([[1,0],[0,0]])
    u = np.array([result,0])
    modes = [mode]
    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))
    
    sigma_A, sigma_AB, sigma_B = chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = chop_in_blocks_vector_multi(means, mode_ind)

    if len(covs) == 1:
        sigma_A = sigma_A[0]
        sigma_B = sigma_B[0]
        sigma_AB = sigma_AB[0]

        #Top left entry of sigma_B: 
        sigma_B = sigma_B[0:1,0:1][0]
        sigma_A_prime = sigma_A - (sigma_B)**(-1)*sigma_AB @ P @ sigma_AB.T

        delta_B = u[np.newaxis,:] - r_B

        r_A_prime = r_A + sigma_B**(-1)* np.einsum("...jk,...k", sigma_AB @ P, delta_B)
        reweights_exp_arg = -0.5*(sigma_B**(-1)*(result - r_B[:,0])**2).reshape(len(log_weights))

        prefactor = 1/np.sqrt(2*np.pi*sigma_B)
    else:
        #Top left entry of sigma_B: 
        sigma_B = sigma_B[:,0:1,0:1]
        sigma_B_inv = (sigma_B)**(-1)
    
        sigma_A_prime = sigma_A - (sigma_B_inv)*sigma_AB @ P[np.newaxis,:,:] @ np.transpose(
            sigma_AB,axes=[0,2,1])

        delta_B = u[np.newaxis,:] - r_B

        r_A_prime = r_A + sigma_B_inv[:,0,0] @ np.einsum("...jk,...k", sigma_AB @ P[np.newaxis,:,:], delta_B)
        reweights_exp_arg = -0.5*sigma_B_inv[:,0,0]*(result - r_B[:,0])**2
        #Norm = np.sqrt(2*np.pi*sigma_B[:,0,0])
    
    
    reweights = log_weights + reweights_exp_arg + np.log(prefactor)
   

    data_A = r_A_prime, sigma_A_prime, reweights, k
    
    return data_A


#Gradients
#-------------------------------------

def project_fock_coherent_gradients(n, data, data_partial, mode, inf=1e-4, hbar = 2):  
    """Returns data tuple after projecting mode on fock state n (in coherent approx)
    
    Args:
        n (int): photon number
        data (tuple) : [means, covs, weights]
        mode (int): mode index that is measured with PNRD
        inf (float): infidelity of the fock approx
        
    Returns:
        data_A (tuple): 
        prob (float): probability of the measurement
    """
    means, covs, log_weights, num_k = data

    if num_k != len(log_weights):
        raise ValueError('Gradients not compatible with reduced Gauss form.')
    
        
    means_f, sigma_f, log_weights_f, k1 = gen_fock_coherent(n, inf, fast=False)
    modes = [mode]

    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))
    
    sigma_A, sigma_AB, sigma_B = chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = chop_in_blocks_vector_multi(means, mode_ind)

    #New array sizes
    M = len(means_f) #Number of weights in POVM
    N = len(r_A) #Number of weights in state
    L = len(r_A[0,:]) #Dimension of output vector (number of modes after measurement)
    
    C = sigma_B + sigma_f
    C_inv = np.linalg.inv(C)

    sigma_ABC = sigma_AB @ C_inv 
    sigma_A_tilde = sigma_ABC @ sigma_AB.transpose(0, 2, 1)
    sigma_A_prime = sigma_A - sigma_A_tilde
    

    delta_B = means_f[np.newaxis,:,:] - r_B[:,np.newaxis,:]
    
    r_A_prime = r_A[:,np.newaxis,:] + np.einsum("...jk,...k", sigma_ABC, delta_B) 

    reweights_exp_arg = -0.5 * np.einsum("...j,...jk,...k", delta_B, C_inv[np.newaxis,:,:], delta_B)

  
    prefactor = np.log(2*np.pi*hbar/np.sqrt(np.linalg.det(2*np.pi*C)))
    
    reweights = log_weights_f[np.newaxis,:] + log_weights[:,np.newaxis] + reweights_exp_arg + prefactor

    reweights = reweights.reshape([M*N])
    r_A_prime = r_A_prime.reshape([M*N,L])


    ### Gradient computation ####

    means_partial, covs_partial, weights_partial = data_partial
    G = covs_partial.shape[0] #Number of gradients
    
    partial_sigma_A, partial_sigma_AB, partial_sigma_B = chop_in_blocks_multi_v2(covs_partial, mode_ind)
    partial_mu_A, partial_mu_B = chop_in_blocks_vector_multi_v2(means_partial, mode_ind)
    
    #Precalculate some stuff

    partial_sigma_B_inv = -C_inv @ partial_sigma_B @ C_inv

    partial_sigma_B_inv_delta_B = np.einsum("...jk,...k", partial_sigma_B_inv[:,np.newaxis,:,:], delta_B[np.newaxis,:,:])
    
    C_inv_delta_B = np.einsum("...jk,...k", C_inv, delta_B)

    #Update the partial derivative of the displacement vector

    partial_AB = np.einsum("...jk,...k", partial_sigma_AB[:,np.newaxis,:,:], C_inv_delta_B[np.newaxis,:,:])
    partial_Cinv = np.einsum("...jk,...k", sigma_AB, partial_sigma_B_inv_delta_B)
    partial_muB = np.einsum("...jk,...k", sigma_ABC, partial_mu_B)

    partial_mu_prime = partial_mu_A[:,:,np.newaxis,:] + partial_AB + partial_Cinv - partial_muB[:,:,np.newaxis,:]
    partial_mu_prime = partial_mu_prime.reshape([G, M*N, L])
  

    #Update the partial derivative of covariance matrix

    partial_sigma_prime = partial_sigma_A + (- partial_sigma_AB @ sigma_ABC.transpose([0,2,1])[np.newaxis,:,:,:]
                                             - sigma_AB[np.newaxis,:,:,:] @ partial_sigma_B_inv @ sigma_AB.transpose([0,2,1])[np.newaxis,:,:,:]
                                             - sigma_ABC[np.newaxis,:,:,:] @ partial_sigma_AB.transpose([0,1,3,2]))

    #Weights of partial derivatives
    partial_muB = np.einsum("...j,...j", partial_mu_B[:,:, np.newaxis,:], C_inv_delta_B[np.newaxis,:,:])

    delta_B_C_inv_delta_B = 0.5 * np.einsum("...j,...j", delta_B[np.newaxis,:,:], partial_sigma_B_inv_delta_B)
   

    new_weights_partial = partial_muB - delta_B_C_inv_delta_B

    prefactor = -0.5 * np.trace(C_inv @ partial_sigma_B, axis1 = 2, axis2 = 3) #Just a number
    
    
    weights_partial_new = weights_partial[:,:,np.newaxis] + new_weights_partial

    weights_partial_new=weights_partial_new.reshape([G, N*M]) + prefactor


    #Return data tuples
    data_A = r_A_prime, sigma_A_prime, reweights, N*M
    data_partial = partial_mu_prime, partial_sigma_prime, weights_partial_new
        
    return (data_A), (data_partial)


    
