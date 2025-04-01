import numpy as np
import bosonicplus

from thewalrus.symplectic import sympmat, xxpp_to_xpxp
from bosonicplus.conversions import dB_to_r, Delta_to_dB

from mpmath import mp

def char_fun(state, alpha, MP = False):
    """
    Single mode characteristic function when state is a sum of Gaussians for hbar = 2
    From Eq. 2.85 of Anders Bjerrum's PhD thesis and Eq. 20 of Weebrook Gaussian QI
    
    L = [Re(alpha), Im(alpha)] 

    Args:
        state : State
        alpha : complex, coordinate
        MP : bool, whether to use higher precision or not
    
    """
    if MP and state.num_k != state.num_weights:
        raise ValueError('mpmath and reduced gaussian method not compatible yet. Use full state description.')
    if state.num_covs != 1:
        raise ValueError('Not sure if function works for states with several covs.')
    means, covs, weights = state.means, state.covs, state.weights

    L = np.array([alpha.real, alpha.imag]) 

    nmodes = state.num_modes

    Omg = xxpp_to_xpxp(sympmat(nmodes)) 
    #Lsymp = Omg.T @ L changing this has no effect
    Lsymp = Omg @ L 

    # Broadcast L 
    L = L[np.newaxis, :]
    Lsymp = Lsymp[np.newaxis,:]
    
    exparg1 = np.einsum("...j,...jk,...k", L, Omg @ covs @ Omg.T, L)
    exparg2 = np.einsum("...jk,...k", means, Lsymp)

    #exparg1 = np.einsum("...j,...jk,...k", L, covs, L)
    #exparg2 = np.einsum("...jk,...k", means, L) #Shape should be (1, num_weights)

    if MP == True:
        charf = np.array([mp.exp(-1/2*exparg1[0]+1j*i) for i in exparg2[0,:]])
        #print(charf)
        #print(weights * charf)
        return mp.fdot(weights, charf)/state.norm
        
    else:
        if state.num_k != state.num_weights: 
            k = state.num_k
            c1 = np.exp(-1/2 * exparg1)
            
            ck1 = np.exp(1j * exparg2[0,0:k])
            ck2 = np.exp(1j * exparg2[0,k:])
            ck3 = np.exp(1j * exparg2[0,k:].conjugate())
            
            charf = c1 * np.sum(weights[0:k]*ck1)
            charf += 0.5 * c1 * np.sum(weights[k:]*ck2)
            charf += 0.5 * c1 * np.sum(weights[k:].conjugate()*ck3)
        else:
            charf = np.exp(-1/2*exparg1[0] + 1j*exparg2[0,:])
            charf = np.sum(weights* charf)
            
        return charf/state.norm
        
def effective_sqz(state, lattice : str, MP = False):
    """Get the effective squeezing of a state according to the Terhal definition

    Args:
        state : State
        lattice: The GKP lattice and direction, sx, sp, rx, rp, hx, hp
        MP: bool, whether to use higher precision from mpmath or not
    """

    lattices = ['sx', 'sp', 'rx', 'rp', 'hx', 'hp']
        
    if lattice not in lattices:
        raise ValueError('Must choose either sx, sp, rx, rp, hx, or hp.')
    
    kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))
    kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))

    #Terhal definition
    if lattice == 'sx':
        alpha = 1j*np.sqrt(np.pi) 
        
    elif lattice == 'sp':        
        alpha = np.sqrt(np.pi)
        
    elif lattice == 'rx':
        alpha = 1j*np.sqrt(2*np.pi)
        
    elif lattice == 'rp':
        alpha = np.sqrt(2*np.pi)
        
    elif lattice == 'hx':
        alpha = 2*np.sqrt(np.pi/np.sqrt(3))  #Grimsmo
        
        alpha = 2*np.sqrt(2*np.pi/np.sqrt(3))*np.exp(-1j*np.pi/6)
        #alpha = 2*(kappa_m +1j*kappa_p)
        
    elif lattice == 'hp':
        #alpha = 2*(kappa_p +1j*kappa_m)
        alpha = 2*np.sqrt(np.pi/np.sqrt(3))*np.exp(1j*2*np.pi/3-np.pi/6*1j)
        #alpha = np.sqrt(2)*np.sqrt(np.pi/np.sqrt(3)/2)


    if MP == True:
        factor = 0.5*(mp.fabs(char_fun(state, alpha, MP=True)) + mp.fabs(char_fun(state, -alpha, MP=True)))
        Delta = complex(mp.sqrt(-2/np.abs(alpha)**2 * mp.log(factor)))
    else:
    
        factor = 0.5*(np.abs(char_fun(state, alpha, MP=False)) + np.abs(char_fun(state, -alpha, MP=False)))
        Delta = np.sqrt(-2/np.abs(alpha)**2 * np.log(factor))

        
    #if MP == True:
        #factor = 0.5*(mp.fabs(char_fun(state, alpha, MP=True)) + mp.fabs(char_fun(state, -alpha, MP=True)))
    
        #factor = float(mp.re(factor))
        #Delta = mp.sqrt(-2/np.abs(alpha)**2 * mp.log(factor))
       # Delta = np.sqrt(-2/np.abs(alpha)**2 * np.log(factor))
        #Delta = mp.re(Delta)
    #factor1 = np.abs(char_fun(state, alpha, MP))
    #factor2 = np.abs(char_fun(state, -alpha, MP))
    #print(factor1, factor2)
    #factor = 0.5*(np.abs(char_fun(state, alpha, MP)) + np.abs(char_fun(state, -alpha, MP)))
    #Delta = np.sqrt(-2/np.abs(alpha)**2 * np.log(factor))
    return Delta

def effective_sqz_old(state, which, MP = True):
    
    if which == 'sx':
        alpha = np.sqrt(np.pi)
        #u = np.sqrt(2*np.pi*sf.hbar)
        u = np.sqrt(2*np.pi)
    elif which == 'sp':
        alpha = 1j*np.sqrt(np.pi)
        #u = np.sqrt(2*np.pi*sf.hbar)
        u = np.sqrt(2*np.pi)
        
    elif which == 'rx':
        alpha = np.sqrt(2*np.pi)
        #u = 2*np.sqrt(np.pi*sf.hbar)
        u = 2*np.sqrt(np.pi)
    elif which == 'rp':
        alpha = 1j*np.sqrt(2*np.pi)
        #u = 2*np.sqrt(np.pi*sf.hbar)
        u = 2*np.sqrt(np.pi)

    
        
    factor = 0.5*(char_fun(state, alpha, MP=MP) + char_fun(state, -alpha, MP=MP))

    if MP:
        #Delta = mp.sqrt(-2/u**2 * mp.log(mp.fabs(factor)))
        Delta = float(mp.re(mp.sqrt(-2/np.abs(alpha)**2 * mp.log(mp.fabs(factor)))))
    else:
        #Delta = np.sqrt(-2/u**2 * np.log(np.abs(factor)))
        Delta = np.sqrt(-2/np.abs(alpha)**2 * np.log(np.abs(factor)))
    return Delta
                           

def Q_expval(state, lattice, N=1, MP = False):
    """Calculate the expectation value of the Q operator defined according to a GKP lattice specified by which
    and according to the grid scaling N
    To do: review against the definitions in states.gkp_squeezing
    
    Args: 
        state : State
        lattice : 0, 1, s0, s1 or h
        N : grid scaling
        MP : bool, whether to use higher precision from mpmath or not 
    Returns:
        expval : float, expectation value of Q operator
    """
    lattices = ['0', '1', 's0', 's1', 'h']
        
    if lattice not in lattices:
        raise ValueError('lattice must be either 0, 1, s0, s1, or h')
        
    if lattice == '0':
        alpha = np.sqrt(2) * np.sqrt(np.pi) * np.sqrt(N)
        expval = 2*char_fun(state, 0, MP) - 1/2 * ( char_fun(state, alpha/2, MP) 
                                                   + char_fun(state,-alpha/2,  MP) 
                                                   + char_fun(state, 1j*alpha, MP) 
                                                   + char_fun(state, -1j*alpha, MP))
                                                                                                    
    elif lattice == '1':
        alpha = np.sqrt(2) * np.sqrt(np.pi) * np.sqrt(N)
        expval = 2*char_fun(state, 0, MP) - 1/2 * ( - char_fun(state, alpha/2, MP) 
                                                   - char_fun(state,-alpha/2, MP) 
                                                   + char_fun(state, 1j*alpha, MP) 
                                                   + char_fun(state, -1j*alpha, MP))
        
    elif lattice == 's0': #symmetric zero
        alpha = np.sqrt(np.pi)* np.sqrt(N)
        expval = 2*char_fun(state, 0, MP) - 1/2 * ( char_fun(state, alpha, MP) 
                                                   + char_fun(state,-alpha, MP) 
                                                   + char_fun(state, 1j*alpha, MP) 
                                                   + char_fun(state, -1j*alpha, MP))
        
    elif lattice == 's1': #symmetric one
        alpha = np.sqrt(np.pi)* np.sqrt(N)
        expval = 2*char_fun(state, 0, MP) - 1/2 * ( -char_fun(state, alpha, MP) 
                                                   -char_fun(state,-alpha, MP) 
                                                   + char_fun(state, 1j*alpha, MP) 
                                                   + char_fun(state, -1j*alpha, MP))
    
    elif lattice == 'h': #hexagonal (does it work?)
        kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))
        kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))
    
        alpha_x = np.sqrt(2)*(kappa_m +1j*kappa_p)* np.sqrt(N)
        alpha_p = np.sqrt(2)*(kappa_p +1j*kappa_m)* np.sqrt(N)
        
        expval = 2*char_fun(state, 0, MP) - 1/2 * ( char_fun(state, alpha_x, MP) 
                                                   + char_fun(state,-alpha_x, MP) 
                                                   + char_fun(state, alpha_p, MP) 
                                                   + char_fun(state, -alpha_p, MP))

    return expval