import numpy as np
from scipy.special import comb
from bosonicplus.states.coherent import gen_fock_coherent
hbar = 2

# Herlper functions from strawberryfields
# ---------------------------------------
def chop_in_blocks_multi(m, id_to_delete):
    r"""
    Splits an array of (symmetric) matrices each into 3 blocks (``A``, ``B``, ``C``).

    Blocks ``A`` and ``C`` are diagonal blocks and ``B`` is the offdiagonal block.

    Args:
        m (ndarray): array of matrices
        id_to_delete (ndarray): array for the indices that go into ``C``

    Returns:
        tuple: tuple of the ``A``, ``B`` and ``C`` matrices
    """
    A = np.delete(m, id_to_delete, axis=1)
    A = np.delete(A, id_to_delete, axis=2)
    B = np.delete(m[:, :, id_to_delete], id_to_delete, axis=1)
    C = m[:, id_to_delete, :][:, :, id_to_delete]
    return (A, B, C)


def chop_in_blocks_vector_multi(v, id_to_delete):
    r"""
    For an array of vectors ``v``, splits ``v`` into two arrays of vectors,
    ``va`` and ``vb``. ``vb`` contains the components of ``v`` specified by
    ``id_to_delete``, and ``va`` contains the remaining components.

    Args:
        v (ndarray): array of vectors
        id_to_delete (ndarray): array for the indices that go into vb

    Returns:
        tuple: tuple of ``(va,vb)`` vectors
    """
    id_to_keep = np.sort(list(set(np.arange(len(v[0]))) - set(id_to_delete)))
    va = v[:, id_to_keep]
    vb = v[:, id_to_delete]
    return (va, vb)




# PNRD MEASUREMENT
# ------------------------------------
def project_fock_coherent(n, data, mode, inf=1e-4):  
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
    means, covs, weights = data

    means_f, sigma_f, weights_f = gen_fock_coherent(n, inf)
    
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

    r_A_tilde = np.einsum("...jk,...k", sigma_ABC, delta_B)
    r_A_prime = r_A[:,np.newaxis,:] + r_A_tilde 
    r_A_prime = r_A_prime.reshape([M*N,L])
    
    reweights_exp_arg = np.einsum("...j,...jk,...k", delta_B, C_inv[np.newaxis,:,:], delta_B)
    #print('Shape reweights_exp_arg' ,reweights_exp_arg.shape)

    Norm = np.exp(-0.5 * reweights_exp_arg) / (np.sqrt(np.linalg.det( 2* np.pi * C))) 
    reweights = weights_f[np.newaxis,:]*weights[:,np.newaxis] * Norm * (2*np.pi*hbar)
    #Norm  = np.sqrt(np.linalg.det( 2* np.pi * C.reshape([M*N,2,2] )))
    reweights = reweights.reshape([M*N])
    prob = np.sum(reweights)
    
    reweights /=  prob

    data_A = r_A_prime, sigma_A_prime, reweights

   
    return data_A, prob


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

    # New data sizes
    M = len(mu_povm)
    N = len(sigma_A)
    L = len(r_A[0,:])

    # Use broadcasting to add arrays of different sizes
    C = (sigma_B[:,np.newaxis,:,:]+sigma_povm[np.newaxis,:,:,:])
    C_inv = np.linalg.inv(C)

    r_A_tilde = np.einsum("...jk,...kl,...l", sigma_AB[:,np.newaxis,:,:], C_inv, r_B[:,np.newaxis,:]) #OBS: mu_povm are zero
    r_A_prime = (r_A[:,np.newaxis,:] - r_A_tilde )
    r_A_prime = r_A_prime.reshape([M*N,L])
    
    sigma_A_tilde = np.einsum("...jk,...kl,...lm",sigma_AB[:,np.newaxis,:,:], C_inv, sigma_AB.transpose(0,2,1)[:,np.newaxis,:,:] )
    sigma_A_prime = (sigma_A[:,np.newaxis,:,:] - sigma_A_tilde)
    sigma_A_prime = sigma_A_prime.reshape([M*N,L,L])

    reweights_exp_arg = np.einsum("...j,...jk,...k", -r_B[:,np.newaxis,:], C_inv, -r_B[:,np.newaxis,:]) #OBS: mu_povm are zero
    Norm = (2*np.pi*hbar) * np.exp(-0.5 * reweights_exp_arg) / (np.sqrt(np.linalg.det( 2* np.pi * C))) 
    new_weights = weights_povm[np.newaxis,:]*weights[:,np.newaxis] * Norm 
    new_weights = new_weights.reshape([M*N])

    prob = np.abs(np.sum(new_weights))
    new_weights /=  prob

    data_A = r_A_prime, sigma_A_prime, new_weights
    
    return data_A, prob 


# Homodyne measurement
# ----------------------------
def project_homodyne(data, mode, result):
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

    sigma_A = sigma_A[0]
    sigma_B = sigma_B[0]
    sigma_AB = sigma_AB[0]

    #Top left entry of sigma_B: 
    sigma_B = sigma_B[0:1,0:1][0]
    sigma_A_prime = sigma_A - (sigma_B)**(-1)*sigma_AB @ P @ sigma_AB.T

    delta_B = u[np.newaxis,:] - r_B

    r_A_tilde = np.einsum("...jk,...k", sigma_AB @ P, delta_B)
    r_A_prime = r_A + sigma_B**(-1)* r_A_tilde 
    
    reweights_exp_arg = (sigma_B**(-1)*(result - r_B[:,0])**2).reshape(len(weights))
    
    reweights_exp = np.exp(-0.5*reweights_exp_arg)
    
    Norm = (reweights_exp / np.sqrt(2*np.pi*sigma_B))
   
    reweights = weights * Norm
    
    prob = np.sum(reweights)

    reweights /=  prob

    data_A = r_A_prime, sigma_A_prime[np.newaxis,:], reweights
    
    return data_A, prob

    
