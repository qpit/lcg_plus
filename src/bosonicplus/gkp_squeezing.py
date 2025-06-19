import numpy as np
from scipy.special import factorial
from bosonicplus.states.fockbasis import density_mn
from bosonicplus.states.coherent import outer_coherent, gen_fock_superpos_coherent
from bosonicplus.charfun import char_fun, char_fun_gradients


def get_gkp_squeezing_stabilizers(lattice, N=1):
    
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


    elif lattice == 'hs0' or lattice == 'hs1': #hexagonal qunaught
        kappa1 = 3**(-1/4) + 3**(1/4)
        kappa2 = 3**(-1/4) - 3**(1/4)
        
        a1 = np.sqrt(np.pi)/2*(kappa1 + 1j*kappa2)* np.sqrt(N)
        a2 = np.sqrt(np.pi)/2*(kappa2 + 1j*kappa1)* np.sqrt(N)

        
    return a1, a2, coeffs
    

def Q_expval(state, lattice, N=1):
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
    a1, a2, coeffs = get_gkp_squeezing_stabilizers(lattice, N)
    
    alphas = [0, a1, -a1, a2, -a2]
    expval = 0
    for i, c in enumerate(coeffs):
        expval += c * char_fun(state, alphas[i])
    
    return expval/2 #Norm wrt Gaussian limit


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

    a1, a2, coeffs = get_gkp_squeezing_stabilizers(lattice, N)
    
    alphas = [0, a1, -a1, a2, -a2]
    expval = 0
    numG = state.covs_partial.shape[0] #number of gradients
    dQ = np.zeros(numG,dtype='complex')
    for i, c in enumerate(coeffs):
        charf, dcharf = char_fun_gradients(state, alphas[i])
        expval += c * charf
        dQ += c * dcharf
    
    return expval/2, dQ/2 #Norm wrt Gaussian limit
    

def gkp_nonlinear_squeezing_operator(cutoff, N=1, lattice = '0'):
    """Construct the GKP nonlinear squeezing operator in the Fock basis up to a cutoff for a type of GKP state
    """
    I = np.eye(cutoff+1, dtype='complex')
    a1, a2, coeffs = get_gkp_squeezing_stabilizers(lattice, N)
    alphas = [0, a1, -a1, a2, -a2]
    rho = 4*I
    for i, c in enumerate(coeffs):
        if i !=0:
            rho += c * density_mn(alphas[i], cutoff)

    return rho /2 #Norm wrt Gaussian limit

def gen_gkp_coherent(n, lattice, N = 1,inf = 1e-4, fast = False):
    """
    Returns state data for 
    Obtain best GKP state in coherent state decomp from the ground state of the GKP nonlinear squeezing operator
    Args: 
        n: Fock cutoff
        which: '0', '1', 's0', 's1', 'h'
        N: scaling of the grid
        inf: (in)fidelity of the coherent state approximation
    """
    rho = gkp_nonlinear_squeezing_operator(n, N, lattice)

    w, v = np.linalg.eigh(rho)
    
    coeffs = v[:,0] #eigs always sorted from lowest to highest eigenvalue, choose lowest
    
    data_gkp = gen_fock_superpos_coherent(coeffs, inf, fast=fast)
  
    
    return data_gkp


def gkp_operator_coherent(cutoff, which, eps):
    """Get GKP non-linear squeezing operator up to a cutoff in the Fock space in the coherent state decomp
    """
    means = []
    log_weights = []
    rho = GKP_nonlinear_squeezing_operator(cutoff, which=which)
    N = cutoff
    coeffs = np.zeros((cutoff+1, cutoff +1), dtype = 'complex')

    for k in np.arange(cutoff+1):
        for l in np.arange(cutoff+1):
            ckl = 0

            for i in np.arange(cutoff+1):
                for j in np.arange(cutoff+1):
                    Aij = rho[i,j]
                    
                    ckl += np.exp(np.abs(eps)**2)/(N+1)**2 * np.sqrt(factorial(i)*factorial(j)*1.0) * Aij/eps**(i+j)*np.exp(-2*np.pi * 1j *(k*i-l*j)/(N+1))
            
            coeffs[k,l] = ckl

            alpha_k = eps*np.exp(2*np.pi*1j*k/(N+1))
            alpha_l = eps*np.exp(2*np.pi*1j*l/(N+1))

            mu, cov, factor = outer_coherent(alpha_k, alpha_l)
            means.append(mu)
            log_weights.append(factor+np.log(ckl))
    
    
            
    return np.array(means), np.array(cov), np.array(log_weights) #Don't normalise! Operator Q is not supposed to be normalised







