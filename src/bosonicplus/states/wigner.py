import numpy as np
from scipy.special import factorial, genlaguerre
from mpmath import mp
from scipy.stats import multivariate_normal
from numba import jit, njit
hbar = 2

def wig_mn(m, n, x, p):
    """Wigner function of |m><n| state
    """
    if n > m:
        m, n = n, m
        p = -p
    
    x /= np.sqrt(hbar)
    p /= np.sqrt(hbar)
    
    return (-1)**n * (x-p*1j)**(m-n) * 1/(hbar * np.pi) * np.exp(-x*x - p*p) * \
            np.sqrt(2**(m-n) * factorial(n) / factorial(m)) * \
            genlaguerre(n, m-n)(2*x*x + 2*p*p)


@njit 
def make_grid(xvec,pvec):
    r"""Returns two coordinate matrices `X` and `P` from coordinate vectors
    `xvec` and `pvec`
    """
    X = np.outer(xvec, np.ones_like(pvec))
    P = np.outer(np.ones_like(xvec), pvec)
    return X,P


def get_wigner_real(data, xvec, pvec):
    """Returns wigner function, but only for real means
    """
    means, covs, weights, norm = data
    X, P = make_grid(xvec,pvec)
    grid = np.empty(X.shape+(2,))
    grid[:, :,0] = X
    grid[:, :,0] = P
    W=0
    for i, mu in enumerate(means):
        if len(covs) == 1:
            mvn  = multivariate_normal(mu.real, covs[0], allow_singular=False) #Only likes real means
        else:
            mvn  = multivariate_normal(mu.real, covs[i], allow_singular=False) #Only likes real means
            
        W += weights[i]*mvn.pdf(grid)
    return W/norm

#Currently unused and slow
def Gauss(sigma, mu, xvec, pvec, MP = False):
    """Returns the Gaussian in phase space point (x,p), or on a grid
    To do: Rethink MP method
    """

    if len(pvec)==1:
        xi  = xvec
    else:
        X, P = make_grid(xvec,pvec)
        xi = np.empty((2,)+X.shape)
        xi[0,:, :] = X
        xi[1,:, :] = P

    sigma_inv = np.linalg.inv(sigma)

    delta = xi - mu[:,np.newaxis, np.newaxis]

    exparg = - 0.5 * np.einsum("j...,...j", delta, np.einsum("...jk,k...",  sigma_inv, delta))

    Norm = 1/np.sqrt(np.linalg.det(sigma*2*np.pi))

    if MP:
        G_mp = np.zeros(exparg.shape, dtype='complex')
        for i in range(exparg.shape[0]):
            for j in range(exparg.shape[1]):
                G_mp[i,j] = mp.fprod([Norm, mp.exp(exparg[i,j])])
        return G_mp
    else: 
        return Norm * np.exp(exparg)  



def Gaussian(sigma, mu, xvec, pvec):
    
    X, P = make_grid(xvec,pvec)
    xi = np.empty((2,)+X.shape)
    xi[0,:, :] = X
    xi[1,:, :] = P
                                    
    delta = xi - mu[:,np.newaxis, np.newaxis]
    sigma_inv= np.linalg.inv(sigma)
    #exparg = -0.5 * (delta @ sigma_inv @ delta)
    exparg = - 0.5 * np.einsum("j...,...j", delta, np.einsum("...jk,k...",  sigma_inv, delta))
    Norm = 1/np.sqrt(np.linalg.det(sigma*2*np.pi))
    return Norm * np.exp(exparg)

