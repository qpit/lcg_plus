import numpy as np
from scipy.special import factorial
from .fockbasis import density_mn
from .coherent import outer_coherent, gen_fock_superpos_coherent

def gkp_nonlinear_squeezing_operator(cutoff, N=1, which = '0'):
    """Construct the GKP nonlinear squeezing operator in the Fock basis up to a cutoff for a type of GKP state
    """
    I = np.eye(cutoff+1)
    
    if which == '0':
        alpha = np.sqrt(2) * np.sqrt(np.pi)*np.sqrt(N)
        
        rho = 1/2 *( 4*I - density_mn(-1j*alpha/2, cutoff) - density_mn(1j*alpha/2,cutoff) - density_mn(alpha,cutoff) - density_mn(-alpha,cutoff))

    elif which == '1':
        alpha = np.sqrt(2) * np.sqrt(np.pi)*np.sqrt(N)
        
        rho = 1/2 *( 4*I + density_mn(-1j*alpha/2, cutoff) + density_mn(1j*alpha/2,cutoff) - density_mn(alpha,cutoff) - density_mn(-alpha,cutoff))

    elif which == 's0': #symmetric zero
        alpha = np.sqrt(np.pi)*np.sqrt(N)
        
        rho = 1/2 *( 4*I - density_mn(-1j*alpha, cutoff) - density_mn(1j*alpha,cutoff) - density_mn(alpha,cutoff) - density_mn(-alpha,cutoff))

    elif which == 's1': #symmetric one
        alpha = np.sqrt(np.pi)*np.sqrt(N)
        
        rho = 1/2 *( 4*I + density_mn(-1j*alpha, cutoff) + density_mn(1j*alpha,cutoff) - density_mn(alpha,cutoff) - density_mn(-alpha,cutoff))

    elif which == 'h': #hexagonal Mareks
        kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))*np.sqrt(N)
        kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))*np.sqrt(N)

        alpha_x = np.sqrt(2)*(kappa_m +1j*kappa_p)
        alpha_p = np.sqrt(2)*(kappa_p +1j*kappa_m)
        
        rho = 1/2 *( 4*I - density_mn(alpha_x, cutoff) - density_mn(-alpha_x,cutoff) - density_mn(alpha_p,cutoff) - density_mn(-alpha_p,cutoff))

    elif which == 'h0':
        #kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))*np.sqrt(N) Petr's 
        #kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))*np.sqrt(N)

        #alpha_x = 2*(kappa_m +1j*kappa_p)
        #alpha_p = kappa_p +1j*kappa_m

        alpha_x = np.sqrt(2)*np.sqrt(2*np.pi/np.sqrt(3))  #Grimsmo
        alpha_p = np.sqrt(np.pi/np.sqrt(3))*np.exp(1j*2*np.pi/3)
        
        rho = 1/2 *( 4*I - density_mn(alpha_x, cutoff) - density_mn(-alpha_x,cutoff) - density_mn(alpha_p,cutoff) - density_mn(-alpha_p,cutoff))
        
    elif which == 'h1':
        #kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))*np.sqrt(N)
        #kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))*np.sqrt(N)

        #alpha_x = 2*(kappa_m +1j*kappa_p)
        #alpha_p = kappa_p +1j*kappa_m

        alpha_x = np.sqrt(2)*np.sqrt(2*np.pi/np.sqrt(3))  #Grimsmo
        alpha_p = np.sqrt(np.pi/np.sqrt(3))*np.exp(1j*2*np.pi/3)
        
        rho = 1/2 *( 4*I - density_mn(alpha_x, cutoff) - density_mn(-alpha_x,cutoff) + density_mn(alpha_p,cutoff) + density_mn(-alpha_p,cutoff))
    return rho

def gkp_operator_coherent(cutoff, which, eps):
    """Get GKP non-linear squeezing operator up to a cutoff in the Fock space in the coherent state decomp
    """
    means = []
    weights = []
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
            weights.append(ckl * factor)
    
    
            
    return np.array(means), np.array(cov), np.array(weights) #Don't normalise! Operator Q is not supposed to be normalised

def gen_gkp_coherent(n, which, N = 1,inf = 1e-4):
    """
    Obtain best GKP state in coherent state decomp from the ground state of the GKP nonlinear squeezing operator
    Args: 
        n: Fock cutoff
        which: '0', '1', 's0', 's1', 'h'
        N: scaling of the grid
        inf: (in)fidelity of the coherent state approximation
    """
    rho = gkp_nonlinear_squeezing_operator(n, N, which)

    w, v = np.linalg.eigh(rho)
    
    coeffs = v[:,0] #eigs always sorted from lowest to highest eigenvalue, choose lowest
    data_gkp = gen_fock_superpos_coherent(coeffs, inf)
    
    return data_gkp


