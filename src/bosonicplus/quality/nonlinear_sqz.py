import numpy as np
from scipy.special import factorial, genlaguerre
from bosonicplus.states.coherent import outer_coherent, gen_fock_superpos_coherent
import strawberryfields as sf

from strawberryfields.backends.states import BaseBosonicState

def Dmn(alpha, n, m):
    """Calculates mn'th element of complex displacement operator with alpha using the Cahill1969 formula
    """
    
    
    if n > m:
        m, n = n, m
        alpha = -alpha.conjugate()

        
    prefactor = np.sqrt(factorial(n)/factorial(m)) * alpha**(m-n) * np.exp(-np.abs(alpha)**2 / 2)
    dmn = prefactor * genlaguerre(n, m-n)(np.abs(alpha)**2)

    return dmn


def density_mn(alpha, cutoff):
    '''
    Construct Fock decomposition of the displacement operator up to a cutoff

    '''
    if cutoff == 0:
        rho = Dmn(alpha, 0,0)
        
    else:
        rho = np.zeros((cutoff+1, cutoff+1), dtype = 'complex')
        
        for i in np.arange(cutoff+1):
            for j in np.arange(cutoff+1):
                rho[i,j] = Dmn(alpha, i,j)
    return rho

def GKP_nonlinear_squeezing_operator(cutoff, N=1, type = '0'):

    '''
    Construct the GKP nonlinear squeezing operator in the Fock basis up to a cutoff for a type of GKP state
    '''

    I = np.eye(cutoff+1)
    
    if type == '0':
        alpha = np.sqrt(2) * np.sqrt(np.pi)*np.sqrt(N)
        
        rho = 1/2 *( 4*I - density_mn(-1j*alpha/2, cutoff) - density_mn(1j*alpha/2,cutoff) - density_mn(alpha,cutoff) - density_mn(-alpha,cutoff))

    elif type == '1':
        alpha = np.sqrt(2) * np.sqrt(np.pi)*np.sqrt(N)
        
        rho = 1/2 *( 4*I + density_mn(-1j*alpha/2, cutoff) + density_mn(1j*alpha/2,cutoff) - density_mn(alpha,cutoff) - density_mn(-alpha,cutoff))

    elif type == 's0': #symmetric zero
        alpha = np.sqrt(np.pi)*np.sqrt(N)
        
        rho = 1/2 *( 4*I - density_mn(-1j*alpha, cutoff) - density_mn(1j*alpha,cutoff) - density_mn(alpha,cutoff) - density_mn(-alpha,cutoff))

    elif type == 's1': #symmetric one
        alpha = np.sqrt(np.pi)*np.sqrt(N)
        
        rho = 1/2 *( 4*I + density_mn(-1j*alpha, cutoff) + density_mn(1j*alpha,cutoff) - density_mn(alpha,cutoff) - density_mn(-alpha,cutoff))

    elif type == 'h': #hexagonal
        kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))*np.sqrt(N)
        kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))*np.sqrt(N)

        alpha_x = np.sqrt(2)*(kappa_m +1j*kappa_p)
        alpha_p = np.sqrt(2)*(kappa_p +1j*kappa_m)
        
        rho = 1/2 *( 4*I - density_mn(alpha_x, cutoff) - density_mn(-alpha_x,cutoff) - density_mn(alpha_p,cutoff) - density_mn(-alpha_p,cutoff))

    elif type == 'h0':
        #kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))*np.sqrt(N) Petr's 
        #kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))*np.sqrt(N)

        #alpha_x = 2*(kappa_m +1j*kappa_p)
        #alpha_p = kappa_p +1j*kappa_m

        alpha_x = np.sqrt(2)*np.sqrt(2*np.pi/np.sqrt(3))  #Grimsmo
        alpha_p = np.sqrt(np.pi/np.sqrt(3))*np.exp(1j*2*np.pi/3)
        
        rho = 1/2 *( 4*I - density_mn(alpha_x, cutoff) - density_mn(-alpha_x,cutoff) - density_mn(alpha_p,cutoff) - density_mn(-alpha_p,cutoff))
        
    elif type == 'h1':
        #kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))*np.sqrt(N)
        #kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))*np.sqrt(N)

        #alpha_x = 2*(kappa_m +1j*kappa_p)
        #alpha_p = kappa_p +1j*kappa_m

        alpha_x = np.sqrt(2)*np.sqrt(2*np.pi/np.sqrt(3))  #Grimsmo
        alpha_p = np.sqrt(np.pi/np.sqrt(3))*np.exp(1j*2*np.pi/3)
        
        rho = 1/2 *( 4*I - density_mn(alpha_x, cutoff) - density_mn(-alpha_x,cutoff) + density_mn(alpha_p,cutoff) + density_mn(-alpha_p,cutoff))
    return rho





# BEST GKP with given fock cutoff in coherent state representation. See GKP-stuff/Mareks_nonlinear_squeezing.ipynb
# -------------------------------------------------------------------------

def get_best_GKP(n, type, N = 1,inf = 1e-4):
    """
    Obtain best GKP state in coherent state decomp from the ground state of the GKP nonlinear squeezing operator
    Args: 
        n: Fock cutoff
        type: '0', '1', 's0', 's1', 'h'
    """
    rho = GKP_nonlinear_squeezing_operator(n, N=N, type = type)

    w, v = np.linalg.eigh(rho)
    
    coeffs = v[:,0] #eigs always sorted from lowest to highest eigenvalue, choose lowest
    data_GKP = gen_fock_superpos_coherent(coeffs, inf)

    gkp = BaseBosonicState(data_GKP, num_modes = 1, num_weights = len(data_GKP[-1]))
    
    return gkp


def Q_operator_coherent(cutoff, type, eps):
    '''
    Get GKP non-linear squeezing operator up to a cutoff in the Fock space in the coherent state decomp
    '''
    means = []
    weights = []
    rho = GKP_nonlinear_squeezing_operator(cutoff, type = type)
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
            weights.append(ckl * factor)
    
    
            
    return np.array(means), np.array(cov), np.array(weights) #Don't normalise! Operator Q is not supposed to be normalised

