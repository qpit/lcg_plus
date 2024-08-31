import numpy as np
from scipy.special import factorial, genlaguerre
import strawberryfields as sf

def wig_mn(m, n, x, p):
    """Wigner function of |m><n| state
    """
    if n > m:
        m, n = n, m
        p = -p
    
    x /= np.sqrt(sf.hbar)
    p /= np.sqrt(sf.hbar)
    
    return (-1)**n * (x-p*1j)**(m-n) * 1/(sf.hbar * np.pi) * np.exp(-x*x - p*p) * \
            np.sqrt(2**(m-n) * factorial(n) / factorial(m)) * \
            genlaguerre(n, m-n)(2*x*x + 2*p*p)


def Gauss(sigma, mu, x, p):
    """Returns the Gaussian in phase space point (x,p), or on a grid
    """

    if len(p)==1:
        xi  = x
    else:
        X, P = np.meshgrid(x,p)
        xi = np.array([X,P])

    sigma_inv = np.linalg.inv(sigma)

    delta = xi - mu[:,np.newaxis, np.newaxis]

    sigma_inv = np.linalg.inv(sigma)

    exparg = - 0.5 * np.einsum("j...,...j", delta, np.einsum("...jk,k...",  sigma_inv, delta))

    Norm = 1/np.sqrt(np.linalg.det(sigma*2*np.pi))
    
    return Norm * np.exp(exparg)

    

def get_wigner_coherent(state, x, p):
    """Returns the Wigner function of the state
    The state must be in the coherent state representation, i.e. there should only be one cov and it should be the vacuum.
    """
    means, cov, weights = state.data

    #Check cov shape (check doesnt work if cov.shape is (2,2) ) 
    #if len(cov) > 1: 
        #raise ValueError('cov is not in the coherent rep. Use state.wigner() instead.')
    
    W = 0
        
    for i, mu in enumerate(means):
        W += weights[i] * Gauss(cov, mu, x, p)
    return W