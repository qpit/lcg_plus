import numpy as np
from thewalrus.symplectic import sympmat, xxpp_to_xpxp
from bosonicplus.conversions import dB_to_r, Delta_to_dB

from mpmath import mp
from scipy.special import logsumexp

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

def effective_sqz_gradients(state, lattice : str):
    """

    Args:
        state : State
        lattice: The GKP lattice and direction, sx, sp, rx, rp, hx, hp
        MP: bool, whether to use higher precision from mpmath or not
    """

    lattices = ['sx', 'sp', 'rx', 'rp', 'hx', 'hp', 'hsx', 'hsp']
        
    if lattice not in lattices:
        raise ValueError('Must choose either sx, sp, rx, rp, hx, or hp.')
    
    kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))
    kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))

    kappa1 = 3**(-1/4) + 3**(1/4)
    kappa2 = 3**(-1/4) - 3**(1/4)

    #Terhal definition
    if lattice == 'sx':
        alpha = 1j*np.sqrt(np.pi) 
        
    elif lattice == 'sp':        
        alpha = np.sqrt(np.pi)
        
    elif lattice == 'rx':
        alpha = 1j*np.sqrt(2*np.pi)
        
    elif lattice == 'rp':
        #alpha = np.sqrt(np.pi/2)
        alpha = np.sqrt(2*np.pi)
        
    elif lattice == 'hx':
        alpha = np.sqrt(np.pi/2) * (kappa1+ 1j*kappa2)
    elif lattice == 'hp':
        #alpha = np.sqrt(np.pi/8) * (kappa2+ 1j*kappa1)
        alpha = np.sqrt(np.pi/2) * (kappa2+ 1j*kappa1)
    elif lattice == 'hsx':
        alpha = np.sqrt(np.pi)/2 * (kappa1 + 1j*kappa2)
    elif lattice == 'hsp': 
        alpha = np.sqrt(np.pi)/2 * (kappa2 + 1j*kappa1)

    f1, df1 = char_fun_gradients(state, alpha)
    f2, df2 = char_fun_gradients(state, -alpha)

    D1 = -2/np.abs(alpha)**2*np.log(np.abs(f1)) #Square of Delta(+alpha)
    D2 = -2/np.abs(alpha)**2*np.log(np.abs(f2)) #Square of Delta(-alpha)


    dD1 =  - 2/np.abs(alpha)**2 * f1 / np.abs(f1)**2 * df1
    dD2 = - 2/np.abs(alpha)**2 * f2 / np.abs(f2)**2 * df2
    
    
    Delta =  0.5 * (np.sqrt(D1)+np.sqrt(D2))
    dDelta =  0.25/np.sqrt(D1)*dD1+0.25/np.sqrt(D2)*dD2
    
    return float(Delta), dDelta


def Q_expval_gradients(state, lattice, N=1):
    """Calculate the expectation value of the Q operator defined according to a GKP lattice specified by which
    and according to the grid scaling N
    To do: review against the definitions in states.gkp_squeezing
    
    Args: 
        state : State
        lattice : 0, 1, s0, s1 or h
        N : grid scaling
    Returns:
        expval : float, expectation value of Q operator
    """
    lattices = ['0', '1', 's0', 's1', 'h0', 'h1', 'hs0', 'hs1']
        
    if lattice not in lattices:
        raise ValueError('lattice must be either 0, 1, s0, s1, h0, h1, hs0 or hs1')


    if '0' in lattice: 
        coeffs = [4, -1, -1, -1, -1]
    elif '1' in lattice:
        coeffs = [4, -1, -1, 1, 1]
        
    if lattice == '0' or lattice =='1': #logical
    
        a1 = np.sqrt(2 * np.pi) * np.sqrt(N)
        a2 = 1j * np.sqrt(np.pi/2) * np.sqrt(N)
                                                                                                        
        
    elif lattice == 's0' or lattice == 's1': #qunaught
        a1 = np.sqrt(np.pi)* np.sqrt(N)
        a2 = 1j*np.sqrt(np.pi)* np.sqrt(N)
        

    elif lattice == 'h0' or lattice == 'h1': #hexagonal logical
        kappa1 = 3**(-1/4) + 3**(1/4)
        kappa2 = 3**(-1/4) - 3**(1/4)
    
        a1 = np.sqrt(np.pi/2)*(kappa1 +1j*kappa2)* np.sqrt(N)
        a2 = np.sqrt(np.pi/8)*(kappa2 +1j*kappa1)* np.sqrt(N)

    elif lattice == 'hs0' or lattice == 'hs1': 
        kappa1 = 3**(-1/4) + 3**(1/4)
        kappa2 = 3**(-1/4) - 3**(1/4)
        
        a1 = np.sqrt(np.pi)/2*(kappa1 + 1j*kappa2)* np.sqrt(N)
        a2 = np.sqrt(np.pi)/2*(kappa2 + 1j*kappa1)* np.sqrt(N)
 
    alphas = [0, a1, -a1, a2, -a2]
    expval = 0
    numG = state.covs_partial.shape[0]
    dQ = np.zeros(numG,dtype='complex')
    for i, c in enumerate(coeffs):
        charf, dcharf = char_fun_gradients(state, alphas[i])
        expval += c * charf
        dQ += c * dcharf
    
    return expval/2, dQ/2 #Norm wrt Gaussian limit
