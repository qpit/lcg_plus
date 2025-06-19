import numpy as np
from thewalrus.symplectic import sympmat, xxpp_to_xpxp
from scipy.special import logsumexp

def char_fun(state, alpha):
    """
    Single mode characteristic function when state is a sum of Gaussians for hbar = 2
    From Eq. 2.85 of Anders Bjerrum's PhD thesis and Eq. 20 of Weebrook Gaussian QI
    
    L = [Re(alpha), Im(alpha)] 

    Args:
        state : State
        alpha : complex, coordinate
    
    """
    
    if state.num_covs != 1:
        raise ValueError('Not sure if function works for states with several covs.')
        
    means, covs, log_weights = state.means, state.covs, state.log_weights

    L = np.array([alpha.real, alpha.imag]) 

    #L = np.array([alpha.imag, alpha.re]) 

    nmodes = state.num_modes

    Omg = xxpp_to_xpxp(sympmat(nmodes)) 
    
    Lsymp = Omg @ L 

    # Broadcast L 
    L = L[np.newaxis, :]
    Lsymp = Lsymp[np.newaxis,:]
    
    exparg1 = -0.5 * np.einsum("...j,...jk,...k", L, Omg @ covs @ Omg.T, L)
    exparg2 = 1j * np.einsum("...jk,...k", means, Lsymp)[0]


    #If only half of the complex means are tracked in the fast rep: 
    if state.num_k != state.num_weights: 
        
        k = state.num_k

        exparg3 = 1j * np.einsum("...jk,...k", means.conjugate(), Lsymp)[0]

        ck1 = exparg2[0:k]
        ck2 = exparg2[k:]
        ck3 = exparg3[k:]

        charfsum = np.concatenate((log_weights[0:k]+ck1, log_weights[k:]+ck2 - np.log(2), np.conjugate(log_weights[k:])+ck3 - np.log(2)))
        
        charf = np.exp(logsumexp(charfsum+exparg1))
        
        return charf/state.norm
    else:
        exparg = exparg1 + exparg2
        
        charf = np.exp(logsumexp(log_weights + exparg))
        return charf/state.norm




def char_fun_gradients(state, alpha):
    """
    Single mode characteristic function when state is a sum of Gaussians for hbar = 2
    From Eq. 2.85 of Anders Bjerrum's PhD thesis and Eq. 20 of Weebrook Gaussian QI
    
    L = [Re(alpha), Im(alpha)] 

    Args:
        state : State
        alpha : complex, coordinate    
    """
    #if MP and state.num_k != state.num_weights:
        #raise ValueError('mpmath and reduced gaussian method not compatible yet. Use full state description.')
    if state.num_covs != 1:
        raise ValueError('Not sure if function works for states with several covs.')
    means, covs, log_weights = state.means, state.covs, state.log_weights

    L = np.array([alpha.real, alpha.imag]) 

    nmodes = state.num_modes

    Omg = xxpp_to_xpxp(sympmat(nmodes)) 
    #Lsymp = Omg.T @ L #changing this has no effect
    Lsymp = Omg @ L 

    # Broadcast L 
    L = L[np.newaxis, :]
    Lsymp = Lsymp[np.newaxis,:]
    
    exparg1 = -0.5 * np.einsum("...j,...jk,...k", L, Omg @ covs @ Omg.T, L)
    exparg2 = np.einsum("...jk,...k", means, Lsymp)[0]
    
    
    exparg = exparg1 + 1j*exparg2

    charfsum = log_weights + exparg
    charf = np.exp(logsumexp(charfsum))

    #### Gradient stuff ####
    means_partial, covs_partial, log_weights_partial = state.means_partial, state.covs_partial, state.log_weights_partial
   
   
    arg1 = 1j * np.einsum("...j,...j", means_partial, Lsymp[np.newaxis,:,:])
    arg2 = -0.5 * np.einsum("...j,...jk,...k", L, Omg @ covs_partial @ Omg.T, L)

    
    charf_partial = log_weights_partial + arg1 + arg2
    
    
    charf_gradient1 = np.exp(logsumexp(charfsum, b = charf_partial, axis = 1))

    #Partial derivative of the norm
    charf_gradient2 = np.exp(logsumexp(log_weights[np.newaxis,:], b = log_weights_partial, axis = 1))

    charf_gradients = charf_gradient1/state.norm - charf/state.norm**2 * charf_gradient2

    return charf/state.norm, charf_gradients