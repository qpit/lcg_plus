import numpy as np
from scipy.special import factorial, genlaguerre
from mpmath import mp
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

def Gauss(sigma, mu, x, p, MP = False):
    """Returns the Gaussian in phase space point (x,p), or on a grid
    To do: Rethink MP method
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

    if MP:
        G_mp = np.zeros(exparg.shape, dtype='complex')
        for i in range(exparg.shape[0]):
            for j in range(exparg.shape[1]):
                G_mp[i,j] = mp.fprod([Norm, mp.exp(exparg[i,j])])[0]
        return G_mp
    else: 
        return Norm * np.exp(exparg)  