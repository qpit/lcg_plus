import numpy as np
from bosonicplus.states.coherent import outer_coherent, gen_indices

def mu_to_alphas(mu):
    """Compute the |alpha><beta| values from the mean
    Args:
        mu : means
    Returns:
        alpha : 
        beta : 
        d : coefficient 
    """
    alpha = 0.5*(mu[:,0].real-mu[:,1].imag)+1j*0.5*(mu[:,0].imag+mu[:,1].real)
    beta = 0.5*(mu[:,0].real+mu[:,1].imag)+1j*0.5*(mu[:,1].real-mu[:,0].imag)
    exparg = -0.5*(mu[:,0].imag)**2 - 0.5*(mu[:,1].imag)**2 + 0.5*1j*(
        mu[:,1].real*mu[:,1].imag+mu[:,0].real*mu[:,0].imag)
    
    d = np.exp(exparg)
    return alpha, beta, d


def reduce(nmax:int, eps:float, data:tuple):
    """Find the weights of the reduced state with stellar rank nmax. The number of new weights is of order (nmax+1)^2.
    

    Args: 
        nmax : max number of photons in reduced state
        eps : radius of target ring
        data : input state parameters
    Returns: 
        new_data : output state parameters
    """
    means, covs, weights, num_k, norm = data
    
    #Normalise weights
    #weights /= norm 
    
    k1 = nmax+1
    
    alpha, beta, d = mu_to_alphas(means)
    
    weights /= d
   
    weights *= np.exp(-0.5 * np.abs(alpha)**2 - 0.5 * np.abs(beta)**2)
    
    w1 = weights[0:num_k][np.newaxis,:]
    w2 = weights[num_k:][np.newaxis,:]/2 #Because we use 2 Re ( weights[num_k:] ) 
    a1 = alpha[0:num_k][np.newaxis,:]
    a2 = alpha[num_k:][np.newaxis,:]
    b2 = beta[num_k:][np.newaxis,:]
    
    ns, ms = gen_indices(nmax)
    n1 = ns[:,np.newaxis]
    m1 = ms[:,np.newaxis]
    n2 = ns[np.newaxis,:]
    m2 = ms[np.newaxis,:]

    #First compute kl sum
    akk = a1**n1 * np.conjugate(a1)**m1
    akl = a2**n1 * np.conjugate(b2)**m1
    alk = b2**n1 * np.conjugate(a2)**m1
 
    sumkk = np.sum(w1*akk,axis=1)
    sumkl = np.sum(w2*akl,axis=1)
    sumlk = np.sum(np.conjugate(w2)*alk,axis=1)
    tot = sumkk + sumkl + sumlk
    
    #Now compute the nm sum for all ij
    n11 = n1[0:k1,:]
    
    exparg1 = -2*np.pi*1j*n11*(n2-m2)/k1
    cnn = 1/eps**(2*n11) * np.exp(exparg1)*tot[0:k1,np.newaxis]

    exparg2 = -2*np.pi*1j*(n1[k1:,:]*n2 - m1[k1:,:]*m2)/k1
    exparg3 = -2*np.pi*1j*(m1[k1:,:]*n2 - n1[k1:,:]*m2)/k1
    
    cnm = 1/eps**(n1[k1:,:]+m1[k1:,:])*np.exp(exparg2)*tot[k1:,np.newaxis]
    cmn = 1/eps**(n1[k1:,:]+m1[k1:,:])*np.exp(exparg3)*np.conjugate(tot[k1:,np.newaxis])
    
    nw1 = np.sum(cnn,axis=0)
    nw2 = np.sum(cnm+cmn,axis=0)
    nw = nw1+nw2

    #Generate the new means, and multiply new weights by the coefficients
      
    theta = 2*np.pi /k1
    alpha = eps * np.exp(1j*theta*ns)
    beta = eps * np.exp(1j*theta*ms)
    means, cov, d = outer_coherent(alpha,beta)

    nw*=d
    nw[k1:]*=2

    #normalise the new weights
    #nw /= np.sum(nw.real)
    
    #return means.T, covs, nw, k1, norm
    return means.T, covs, nw, k1, np.sum(nw.real)