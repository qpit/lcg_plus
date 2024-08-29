import strawberryfields as sf
import numpy as np
from scipy.special import comb
from strawberryfields.backends.bosonicbackend import ops
#from strawberryfields.backends.bosonicbackend.bosoniccircuit import BosonicModes

from bosonicplus.states.coherent import gen_fock_coherent
#from bosonicplus.ng_states import gen_fock_coherent, eps_fock_coherent

# Different conventions in these codes (REVIEW)

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

    #Given infidelity, find a suitable alpha for fock_coherent
    #alpha = eps_fock_coherent(n, inf)
    
    means_f, sigma_f, weights_f = gen_fock_coherent(n, inf)
    
    modes = [mode]

    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))
    
    sigma_A, sigma_AB, sigma_B = ops.chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = ops.chop_in_blocks_vector_multi(means, mode_ind)
    
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
    reweights = weights_f[np.newaxis,:]*weights[:,np.newaxis] * Norm * (2*np.pi*sf.hbar)
    #Norm  = np.sqrt(np.linalg.det( 2* np.pi * C.reshape([M*N,2,2] )))
    reweights = reweights.reshape([M*N])
    prob = np.sum(reweights)
    
    reweights /=  prob

    data_A = r_A_prime, sigma_A_prime, reweights

   
    return data_A, prob

def post_select_fock_coherent(circuit, mode, n, inf = 1e-4, out = False):
    """Post select on counting n photons in mode. ?New circuit has one less mode, so be careful with indexing.

    Args: 
        circuit (BosonicModes): circuit object
        mode (int) : measured mode index 
        n (int) : photon number

    Returns: updates circuit object
    """

    if sf.hbar != 2:
        raise ValueError('Due to how state data is stored in BosonicModes and BaseBosonicState, setting sf.hbar!=2 will give wrong results.')

    data = circuit.means, circuit.covs, circuit.weights

    data_out, prob = project_fock_coherent(n, data, mode, inf)

    if out:
        print(f'Measuring {n} photons in mode {mode}.')
        print(f'Data shape before measurement, {[i.shape for i in data]}.')
        print('Probability of measurement = {:.3e}'.format(prob))
        print(f'Data shape after measurement, {[i.shape for i in data_out]}')
    
    
    # Delete the measured mode 
    num_modes = len(circuit.get_modes())
    circuit.reset(num_subsystems = num_modes -1)
    
    #Update circuit data
    means, cov, weights = data_out
    circuit.covs = cov
    circuit.means = means
    circuit.weights = weights
    circuit.success *= prob


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
    
    covs = np.array([np.eye(2)*(1+2*nbar) for nbar in nbars])
    means = np.repeat( np.zeros(2)[np.newaxis,:], k+1, axis = 0)

    data = means, covs, weights
    
    return data


def project_ppnrd_thermal(data, mode, n, M):
    
    means, covs, weights = data
    
    modes = [mode]
    mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1))

    sigma_A, sigma_AB, sigma_B = ops.chop_in_blocks_multi(covs, mode_ind)
    r_A, r_B = ops.chop_in_blocks_vector_multi(means, mode_ind)

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
    Norm = (2*np.pi*sf.hbar) * np.exp(-0.5 * reweights_exp_arg) / (np.sqrt(np.linalg.det( 2* np.pi * C))) 
    new_weights = weights_povm[np.newaxis,:]*weights[:,np.newaxis] * Norm 
    new_weights = new_weights.reshape([M*N])

    prob = np.abs(np.sum(new_weights))
    new_weights /=  prob

    data_A = r_A_prime, sigma_A_prime, new_weights
    
    return data_A, prob 


def post_select_ppnrd_thermal(circuit, mode, n, M, out =False):
    """
    Detect mode wth pPNRD registering n clicks by demultiplexing into M on/off detectors.
    The pPNRD POVM is written as a linear combination of Gaussians (thermal states) and the
    circuit's Gaussian means, covs and weights are updated according to the Gaussian transformation rules of 
    Bourassa et al. 10.1103/PRXQuantum.2.040315 . 

    Extension/generalisation of code from strawberryfield's bosonicbackend. 
    
    To do: 
        Write down formula in documentation.
    
    Args: 
        circuit (object): BosonicModes class
        mode (int): mode to be detected
        n (int): number of clicks detected
        M (int): number of on/off detectors in the click-detector    
        out (bool): print output text
        
    Returns: updates circuit object
    
    """
    if n > M:
        raise ValueError('Number of clicks cannot exceed click detectors.')

    if sf.hbar != 2:
        raise ValueError('Due to how state data is stored in BosonicModes and BaseBosonicState, setting sf.hbar!=2 will give wrong results.')

    # Extract circuit data
    data = circuit.means, circuit.covs, circuit.weights

    data_out, prob = project_ppnrd_thermal(data, mode, n, M)

    if out:
        print(f'Measuring {n} clicks in mode {mode}.')
        print(f'Data shape before measurement, {[i.shape for i in data]}.')
        print('Probability of measurement = {:.3e}'.format(prob))
        print(f'Data shape after measurement, {[i.shape for i in data_out]}')
        
    # Delete the measured mode 
    num_modes = len(circuit.get_modes())
    circuit.reset(num_subsystems = num_modes -1)
    
    #Update circuit data
    means, covs, weights = data_out
    circuit.covs = covs
    circuit.means = means
    circuit.weights = weights
    circuit.success *= prob
    
