import numpy as np
from math import factorial
from lcg_plus.base import State
from .coherent import gen_fock_coherent, gen_sqz_cat_coherent, gen_fock_coherent_old
from lcg_plus.gkp_squeezing import gen_gkp_coherent
from thewalrus.symplectic import xxpp_to_xpxp, squeezing, beam_splitter


def prepare_fock_coherent(n, inf=1e-4, epsilon = None, fast = True):
    """Prepare Fock state in coherent state approx"""
    if fast: 
        data = gen_fock_coherent(n, inf, epsilon, norm =True)
    else:
        data = gen_fock_coherent_old(n, inf, epsilon)
        
    fock = State(1)
    fock.update_data(data)
    return fock

def prepare_sqz_cat_coherent(r, alpha, k, fast = False):
    """Prepare a squeezed cat, requires a higher precision with mp.math
    Args: 
        r : squeezing of the cat
        alpha: displacement of the cat (pre-squeezing)
        k : parity
    Returns:
        State
    
    """
    data = gen_sqz_cat_coherent(r, alpha, k, fast)
    
    sq_cat = State(1)
    sq_cat.update_data(data)
    return sq_cat

def prepare_gkp_nonlinear_sqz(n, lattice, N = 1, inf = 1e-4, fast=True):
    """
    Obtain best GKP state in coherent state decomp from the ground state of the GKP nonlinear squeezing operator
    Args: 
        n: Fock cutoff
        lattice: '0', '1', 's0', 's1', 'h0', 'h1', 'hs0', 'hs1'
        N: scaling of the grid
        inf: infidelity of the coherent state approximation
    Returns:
        bosonicplus.base.State
    """
    
    data_gkp = gen_gkp_coherent(n, lattice, N, inf, fast)

        
    state = State(1)
    state.update_data(data_gkp)
    
    return state

def prepare_phssv(r, T=0.99, eta=1):
    """Prepare a lossy photon subtracted single mode squeezed vacuum state
    by splitting the state on a 99:1 beam splitter, and heralding on a click 
    of the ancillary vacuum mode. Apply pure loss channel to output state.
    Args: 
        r : squeezing
        T : transmissivity of beamsplitter (0.99) 
        eta : transmittivity of the loss channel
    Returns:
        bosonicplus.base.State
    """
    state = State(1)
    #Squeeze the state in mode 0
    state.apply_symplectic(xxpp_to_xpxp(squeezing(r)))
    
    #Add ancilla vacuum
    state.add_state(State(1))
    
    #99:1 beamsplitter
    state.apply_symplectic(xxpp_to_xpxp(beam_splitter(np.arccos(np.sqrt(T)),0)))
    
    #Measure a click in mode 1
    state.post_select_ppnrd_thermal(1,1,1) 

    #pure loss channel
    state.apply_loss(np.array(eta),np.array([0]))
    return state

def gkp_pauli_operator(k : int, eps : float,cutoff=1e-10, hbar = 1):
    """Generate the coefficients, means and covariance of the GKP Pauli operator given in 
    the Appendix of https://doi.org/10.1103/PhysRevA.108.052413

    I've swapped x and p compared to his def. 

    The sum of Gaussians is divided into a sum of Gaussians in x times a sum of Gaussians in p

    Args:
        k : which Pauli operator
        eps : fock damping strength
        cutoff : max coefficient value

    Returns:
        cx, mux : coefficients and means in x 
        cp, mup : coefficients and means in p 
        cov : (shared) covariance mat
        
    """
    
    cov = hbar/2*np.tanh(eps)*np.eye(2)
    m_max = int(np.sqrt(-np.log(cutoff)*4/np.pi/np.tanh(eps)))
    
    if k == 0:
        s1 = 1
        s2 = 1
        #Range of even integers values, m1, m2 even
        if m_max%2 == 1: 
            m1 = np.arange(-m_max-1,m_max+2,2)
            m2 = np.arange(-m_max-1,m_max+2,2)
        elif m_max %2 == 0:
            m1 = np.arange(-m_max,m_max+2,2)
            m2 = np.arange(-m_max,m_max+2,2)

    elif k == 1:
        s1 = 1
        #m1 odd, m2 even
        if m_max%2 == 1:
            m1 = np.arange(-m_max,m_max+2,2)
            m2 = np.arange(-m_max-1,m_max+2,2)
        elif m_max %2 == 0:
            m1 = np.arange(-m_max-1,m_max+2,2)
            m2 = np.arange(-m_max,m_max+2,2)
        s2 = (-1)**(m2/2)
        
    elif k == 2:
        #m1 odd, m2 odd
        if m_max%2 == 1:
            m1 = np.arange(-m_max,m_max+2,2,dtype=complex)
            m2 = np.arange(-m_max,m_max+2,2,dtype=complex)
        elif m_max%2 == 0:
            m1 = np.arange(-m_max-1,m_max+2,2,dtype=complex)
            m2 = np.arange(-m_max-1,m_max+2,2,dtype=complex)
            
        
        s1 = (-1)**(m1/2)
        s2 = -(-1)**(m2/2)
    elif k == 3:
        s2 = 1
        #m1 even, m2 odd
        if m_max%2 == 1:
            m1 = np.arange(-m_max-1,m_max+2,2)
            m2 = np.arange(-m_max,m_max+2,2,dtype=complex)
        elif m_max%2 == 0:
            m1 = np.arange(-m_max,m_max+2,2)
            m2 = np.arange(-m_max-1,m_max+2,2,dtype=complex)
            
        s1 = (-1)**(m1/2)
            
    cx = np.exp(-np.tanh(eps)*np.pi/4*m2**2)*s2
    cp = np.exp(-np.tanh(eps)*np.pi/4*m1**2)*s1

    mux = 1/np.cosh(eps)*np.sqrt(np.pi/2)*np.sqrt(hbar/2)*m2
    mup = 1/np.cosh(eps)*np.sqrt(np.pi/2)*np.sqrt(hbar/2)*m1

    return cx, mux, cp, mup, cov
    