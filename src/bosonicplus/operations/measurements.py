import numpy as np
from scipy.special import comb
from bosonicplus.states.coherent import gen_fock_coherent, gen_fock_bosonic, gen_fock_coherent_old
from bosonicplus.from_sf import chop_in_blocks_multi, chop_in_blocks_vector_multi
from mpmath import mp
hbar = 2


# PNRD MEASUREMENT
# ------------------------------------
def project_fock_coherent(n, data, mode, inf=1e-4, k2=None):  
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
    means, covs, weights = data
    if k2: 
        means_f, sigma_f, weights_f, k1, norm = gen_fock_coherent(n, inf, fast=True)
    else:
        means_f, sigma_f, weights_f, k1, norm = gen_fock_coherent_old(n, inf)
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

    reweights_exp_arg = np.einsum("...j,...jk,...k", delta_B, C_inv[np.newaxis,:,:], delta_B)
    #print('Shape reweights_exp_arg' ,reweights_exp_arg.shape)

    Norm = np.exp(-0.5 * reweights_exp_arg) / (np.sqrt(np.linalg.det( 2* np.pi * C)))
    reweights = weights_f[np.newaxis,:]*weights[:,np.newaxis] * Norm * (2*np.pi*hbar)

    if k2 and k2>1:
        
        delta_B_special = means_f[np.newaxis,k1:] - np.conjugate(r_B[k2:,np.newaxis,:])
        r_A_prime_special = np.conjugate(r_A[k2:,np.newaxis,:]) + np.einsum("...jk,...k", sigma_ABC, delta_B_special)
        reweights_exp_arg_special = np.einsum("...j,...jk,...k", delta_B_special, C_inv[np.newaxis,:,:], delta_B_special)
        Norm_special = np.exp(-0.5 * reweights_exp_arg_special) / (np.sqrt(np.linalg.det( 2* np.pi * C)))
        reweights_special = weights_f[np.newaxis,k1:]*np.conjugate(weights[k2:,np.newaxis]) * Norm_special * (2*np.pi*hbar)

        #Building the new data tuple with the k1*k2 normal gaussians appearing first
        
        w1 = reweights[0:k2,0:k1].flatten()
        w2 = reweights[0:k2,k1:].flatten()
        w3 = reweights[k2:,0:k1].flatten()
        w4 = 0.5*reweights[k2:,k1:].flatten()
        w5 = 0.5*reweights_special.flatten()
        
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
        prob = np.sum(reweights.real)
        data_A = r_A_prime, sigma_A_prime, reweights, k1*k2, prob
        
        return data_A
    else:
        prob=np.sum(reweights)
        data_A = r_A_prime, sigma_A_prime, reweights, len(reweights), prob
        
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


def ppnrd_povm_thermal(k, N):
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
        eta[0] = 1e-15 #close to zero to avoid inf
    
    nbars = (1 - eta)/eta
    weights = comb(N,k) * comb(k,ls)* (-1)**ls * 1/eta 
    
    covs = np.array([np.eye(2)*hbar/2*(1+2*nbar) for nbar in nbars])
    means = np.repeat( np.zeros(2)[np.newaxis,:], k+1, axis = 0)

    data = means, covs, weights
    
    return data

def project_ppnrd_thermal(data, mode, n, M):
    
    means, covs, weights = data
    
    modes = [mode]
    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))

    sigma_A, sigma_AB, sigma_B = chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = chop_in_blocks_vector_multi(means, mode_ind)

    # pPNRD PNRD POVM with thermal states
    mu_povm, sigma_povm, weights_povm = ppnrd_povm_thermal(n, M)    

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

    reweights_exp_arg = np.einsum("...j,...jk,...k", -r_B[:,np.newaxis,:], C_inv, -r_B[:,np.newaxis,:]) #OBS: mu_povm are zero
    Norm = (2*np.pi*hbar) * np.exp(-0.5 * reweights_exp_arg) / (np.sqrt(np.linalg.det( 2* np.pi * C))) 
    new_weights = weights_povm[np.newaxis,:]*weights[:,np.newaxis] * Norm 
    new_weights = new_weights.reshape([M*N])

    prob = np.sum(new_weights)
    #new_weights /=  prob

    data_A = r_A_prime, sigma_A_prime, new_weights, len(new_weights), prob
    
    return data_A 

def project_fock_thermal(data, mode, n ,r = 0.05):
    means, covs, weights = data
    
    modes = [mode]
    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))

    sigma_A, sigma_AB, sigma_B = chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = chop_in_blocks_vector_multi(means, mode_ind)

    # Fock state with thermal states
    mu_povm, sigma_povm, weights_povm = gen_fock_bosonic(n, r)
   
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

    reweights_exp_arg = np.einsum("...j,...jk,...k", -r_B[:,np.newaxis,:], C_inv, -r_B[:,np.newaxis,:]) #OBS: mu_povm are zero
    Norm = (2*np.pi*hbar) * np.exp(-0.5 * reweights_exp_arg) / (np.sqrt(np.linalg.det( 2* np.pi * C))) 
    new_weights = weights_povm[np.newaxis,:]*weights[:,np.newaxis] * Norm 
    new_weights = new_weights.reshape([M*N])

    prob = np.sum(new_weights)
    #new_weights /=  prob

    data_A = r_A_prime, sigma_A_prime, new_weights, len(new_weights), prob
    return data_A


# Homodyne measurement
# ----------------------------
def project_homodyne(data, mode, result, MP = False):
    r"""Following Brask's Gaussian note (single mode)
    Do a homodyne x-measurement on one mode
    """

    means, covs, weights = data
    
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
        reweights_exp_arg = (sigma_B**(-1)*(result - r_B[:,0])**2).reshape(len(weights))

        Norm = np.sqrt(2*np.pi*sigma_B)
    else:
        #Top left entry of sigma_B: 
        sigma_B = sigma_B[:,0:1,0:1]
        sigma_B_inv = (sigma_B)**(-1)
        print(sigma_B_inv.shape)
    
        sigma_A_prime = sigma_A - (sigma_B_inv)*sigma_AB @ P[np.newaxis,:,:] @ np.transpose(
            sigma_AB,axes=[0,2,1])

        delta_B = u[np.newaxis,:] - r_B

        r_A_prime = r_A + sigma_B_inv[:,0,0] @ np.einsum("...jk,...k", sigma_AB @ P[np.newaxis,:,:], delta_B)
        reweights_exp_arg = sigma_B_inv[:,0,0]*(result - r_B[:,0])**2
        Norm = np.sqrt(2*np.pi*sigma_B[:,0,0])
    
        
    if MP: 
        reweights_exp = np.array([mp.exp(-0.5*i) for i in reweights_exp_arg])
    else:
        reweights_exp = np.exp(-0.5*reweights_exp_arg)
    
    reweights = weights*reweights_exp/ Norm #mp?
    
    if MP:
        prob = mp.fdot(weights, Norm )
    else:
        prob = np.sum(reweights)

    data_A = r_A_prime, sigma_A_prime, reweights, len(reweights), prob
    
    return data_A


    
