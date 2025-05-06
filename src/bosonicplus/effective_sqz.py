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
    #if MP and state.num_k != state.num_weights:
        #raise ValueError('mpmath and reduced gaussian method not compatible yet. Use full state description.')
    if state.num_covs != 1:
        raise ValueError('Not sure if function works for states with several covs.')
    means, covs, weights = state.means, state.covs, state.weights

    L = np.array([alpha.real, alpha.imag]) 

    nmodes = state.num_modes

    Omg = xxpp_to_xpxp(sympmat(nmodes)) 
    #Lsymp = Omg.T @ L #changing this has no effect
    Lsymp = Omg @ L 

    # Broadcast L 
    L = L[np.newaxis, :]
    Lsymp = Lsymp[np.newaxis,:]
    
    exparg1 = np.einsum("...j,...jk,...k", L, Omg @ covs @ Omg.T, L)
    exparg2 = np.einsum("...jk,...k", means, Lsymp)   
    
    if state.num_k != state.num_weights: 
        k = state.num_k
        if MP:
            #print('exparg1', exparg1[0])
            #c1 = mp.exp(-1/2 * exparg1[0])
            #print('c1', c1)
            #ck1 = np.array([mp.exp(1j * i) for i in exparg2[0,0:k]])
            #print('ck1', ck1)
            #ck2 = np.array([mp.exp(1j * i) for i in exparg2[0,k:]])
            #print('ck2', ck2)
            #ck3 = np.array([mp.exp(1j * i) for i in exparg2[0,k:].conjugate()])
            #print('ck3', ck3)

            #charf = c1 * mp.fdot(weights[0:k], ck1)
            #print('cf',charf)
            #charf += 0.5* c1 * mp.fdot(weights[k:], ck2)
            #print('cf2',c1 * mp.fdot(weights[k:], ck2))
            #charf += 0.5* c1 * mp.fdot(weights[k:].conjugate() ,ck3)
            #print('cf3',c1 * mp.fdot(weights[k:].conjugate(), ck3))
            
            
            charf = np.array([mp.exp(-1/2*exparg1[0] + 1j*i) for i in exparg2[0,:]])
            return mp.re(mp.fdot(weights, charf))/state.norm.real     
            
        else:
            
            c1 = np.exp(-1/2 * exparg1)
            ck1 = np.exp(1j * exparg2[0,0:k])
            ck2 = np.exp(1j * exparg2[0,k:])
            ck3 = np.exp(1j * exparg2[0,k:].conjugate())
            
            charf = c1 * np.sum(weights[0:k]*ck1)
            charf += c1 * np.sum(weights[k:]*ck2)
            #charf += 0.5 * c1 * np.sum(weights[k:]*ck2)
            #charf += 0.5 * c1 * np.sum(weights[k:].conjugate()*ck3)
            #charf = np.exp(-1/2*exparg1[0] + 1j*exparg2[0,:])
            #charf = np.sum(weights* charf)
            return charf/state.norm
    else:
        if MP:
            charf = np.array([mp.exp(-1/2*exparg1[0] + 1j*i) for i in exparg2[0,:]])
            return mp.fdot(weights, charf)/state.norm     
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

    f1 = char_fun(state, alpha, MP)
    f2 = char_fun(state, -alpha, MP)

    if MP == True:
        
        
        D1 = mp.sqrt(-2/np.abs(alpha)**2*mp.log(mp.fabs(f1)))
        D2 = mp.sqrt(-2/np.abs(alpha)**2*mp.log(mp.fabs(f2)))
        Delta = 0.5 * (D1+D2)
    else:
        
        D1 = np.sqrt(-2/np.abs(alpha)**2*np.log(np.abs(f1)))
        D2 = np.sqrt(-2/np.abs(alpha)**2*np.log(np.abs(f2)))
        
        Delta = 0.5 * (D1+D2)
    

        
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
    return float(Delta)

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

def Q_expval1(state, type, N=1, MP = False):
    
    if type == '0':
        alpha = np.sqrt(2) * np.sqrt(np.pi) * np.sqrt(N)
        expval = 2*char_fun(state, 0, MP) - 1/2 * ( char_fun(state, alpha/2, MP) + char_fun(state,-alpha/2,  MP) + char_fun(state, 1j*alpha, MP) + char_fun(state, -1j*alpha, MP))
                                                                                                    
    elif type == '1':
        alpha = np.sqrt(2) * np.sqrt(np.pi) * np.sqrt(N)
        expval = 2*char_fun(state, 0, MP) - 1/2 * ( - char_fun(state, alpha/2, MP) - char_fun(state,-alpha/2, MP) + char_fun(state, 1j*alpha, MP) + char_fun(state, -1j*alpha, MP))
        
    elif type == 's0': #symmetric zero
        alpha = np.sqrt(np.pi)* np.sqrt(N)
        expval = 2*char_fun(state, 0, MP) - 1/2 * ( char_fun(state, alpha, MP) + char_fun(state,-alpha, MP) + char_fun(state, 1j*alpha, MP) + char_fun(state, -1j*alpha, MP))
        
    elif type == 's1': #symmetric one
        alpha = np.sqrt(np.pi)* np.sqrt(N)
        expval = 2*char_fun(state, 0, MP) - 1/2 * ( -char_fun(state, alpha, MP) -char_fun(state,-alpha, MP) + char_fun(state, 1j*alpha, MP) + char_fun(state, -1j*alpha, MP))
    
    elif type == 'h': #hexagonal
        kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))
        kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))
    
        alpha_x = np.sqrt(2)*(kappa_m +1j*kappa_p)* np.sqrt(N)
        alpha_p = np.sqrt(2)*(kappa_p +1j*kappa_m)* np.sqrt(N)
        
        expval = 2*char_fun(state, 0, MP) - 1/2 * ( char_fun(state, alpha_x, MP) + char_fun(state,-alpha_x, MP) + char_fun(state, alpha_p, MP) + char_fun(state, -alpha_p, MP))

    return expval.real

                           

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
    lattices = ['0', '1', 's0', 's1', 'h0', 'h1', 'hs0', 'hs1']
        
    if lattice not in lattices:
        raise ValueError('lattice must be either 0, 1, s0, s1, h0, h1, hs0 or hs1')

    if lattice == '0':
    
        alphax = np.sqrt(2 * np.pi) * np.sqrt(N)
        alphap = np.sqrt(np.pi/2) * np.sqrt(N)

        expval = 4*char_fun(state, 0, MP) -  ( char_fun(state, alphax, MP) 
                                                   + char_fun(state,-alphax,  MP) 
                                                   + char_fun(state, 1j*alphap, MP) 
                                                   + char_fun(state, -1j*alphap, MP))
                                                                                                    
    elif lattice == '1':
        alphax = np.sqrt(2 * np.pi) * np.sqrt(N)
        alphap = np.sqrt(np.pi/2) * np.sqrt(N)
        
        expval = 4*char_fun(state, 0, MP) - ( char_fun(state, alphax, MP) 
                                                   +char_fun(state,-alphax, MP) 
                                                   - char_fun(state, 1j*alphap, MP) 
                                                   - char_fun(state, -1j*alphap, MP))
        
    elif lattice == 's0': #symmetric zero
        alpha = np.sqrt(np.pi)* np.sqrt(N)
        expval = 4*char_fun(state, 0, MP) -  ( char_fun(state, alpha, MP) 
                                                   + char_fun(state,-alpha, MP) 
                                                   + char_fun(state, 1j*alpha, MP) 
                                                   + char_fun(state, -1j*alpha, MP))
        
    elif lattice == 's1': #symmetric one
        alpha = np.sqrt(np.pi)* np.sqrt(N)
        expval = 4*char_fun(state, 0, MP) - ( char_fun(state, alpha, MP) 
                                                   + char_fun(state,-alpha, MP) 
                                                   - char_fun(state, 1j*alpha, MP) 
                                                   - char_fun(state, -1j*alpha, MP))

    elif lattice == 'h0': 
        kappa1 = 3**(-1/4) + 3**(1/4)
        kappa2 = 3**(-1/4) - 3**(1/4)
    
        gamma = np.sqrt(np.pi/2)*(kappa1 +1j*kappa2)* np.sqrt(N)
        delta = np.sqrt(np.pi/8)*(kappa2 +1j*kappa1)* np.sqrt(N)
        
        expval = 4*char_fun(state, 0, MP) - ( char_fun(state, gamma, MP) 
                                                   + char_fun(state,-gamma, MP) 
                                                   + char_fun(state, delta, MP) 
                                                   + char_fun(state, -delta, MP))

    elif lattice == 'h1': 
        kappa1 = 3**(-1/4) + 3**(1/4)
        kappa2 = 3**(-1/4) - 3**(1/4)
    
        gamma = np.sqrt(np.pi/2)*(kappa1 + 1j*kappa2)* np.sqrt(N)
        delta = np.sqrt(np.pi/8)*(kappa2 + 1j*kappa1)* np.sqrt(N)
        
        expval = 4*char_fun(state, 0, MP) - ( char_fun(state, gamma, MP) 
                                                   + char_fun(state,-gamma, MP) 
                                                   - char_fun(state, delta, MP) 
                                                   - char_fun(state, -delta, MP))

    elif lattice == 'hs1': 
        kappa1 = 3**(-1/4) + 3**(1/4)
        kappa2 = 3**(-1/4) - 3**(1/4)
        
        gamma = np.sqrt(np.pi)/2*(kappa1 + 1j*kappa2)* np.sqrt(N)
        delta = np.sqrt(np.pi)/2*(kappa2 + 1j*kappa1)* np.sqrt(N)
        
        expval = 4*char_fun(state, 0, MP) - ( char_fun(state, gamma, MP) 
                                                   + char_fun(state,-gamma, MP) 
                                                   - char_fun(state, delta, MP) 
                                                   - char_fun(state, -delta, MP))

    elif lattice == 'hs0': 
        kappa1 = 3**(-1/4) + 3**(1/4)
        kappa2 = 3**(-1/4) - 3**(1/4)
    
        gamma = np.sqrt(np.pi)/2*(kappa1 + 1j*kappa2)* np.sqrt(N)
        delta = np.sqrt(np.pi)/2*(kappa2 + 1j*kappa1)* np.sqrt(N)
        
        expval = 4*char_fun(state, 0, MP) - ( char_fun(state, gamma, MP) 
                                                   + char_fun(state,-gamma, MP) 
                                                   + char_fun(state, delta, MP) 
                                                   + char_fun(state, -delta, MP))
    
    elif lattice == 'hpetr': #hexagonal (does it work?)
        kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))
        kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))
    
        alpha_x = np.sqrt(2)*(kappa_m +1j*kappa_p)* np.sqrt(N)
        alpha_p = np.sqrt(2)*(kappa_p +1j*kappa_m)* np.sqrt(N)
        
        expval = 4 * char_fun(state, 0, MP) - ( char_fun(state, alpha_x, MP) 
                                                   + char_fun(state,-alpha_x, MP) 
                                                   + char_fun(state, alpha_p, MP) 
                                                   + char_fun(state, -alpha_p, MP))

    return expval/2 #Norm wrt Gaussian limit