# COHERENT STATE SIMULATION MODULE
# ----------------------------------
# Simulation tools for approximating Fock state superpositions as
# superpositions of coherent states from Marshall & Anand http://arxiv.org/abs/2305.17099.

# The Wigner function of the superposition of coherent states is a weighted sum of Gaussians.

import numpy as np
from math import factorial
from mpmath import mp, fp 
from scipy.special import comb
#from bosonicplus.base import State
hbar = 2

def outer_coherent(alpha, beta):
    """ Returns the coefficient, displacement vector and covariance matrix (vacuum) of the Gaussian that
    describes the Wigner function of the outer product of two coherent states |alpha><beta| derived 
    in Appendix A of https://arxiv.org/abs/2103.05530.
    """
    cov = hbar/2 * np.eye(2)
    re_alpha = alpha.real
    im_alpha = alpha.imag
    re_beta = beta.real
    im_beta = beta.imag

    mu = np.sqrt(hbar/2) * np.array([re_alpha + re_beta 
                                        + 1j *(im_alpha - im_beta), 
                                        im_alpha + im_beta
                                        + 1j * (re_beta - re_alpha)])
    coeff = np.exp( -0.5 * (im_alpha - im_beta) **2
                   - 0.5 * (re_alpha - re_beta) **2
                   - 1j * im_beta * re_alpha + 1j * im_alpha * re_beta)

    return mu, cov, coeff

def eps_fock_coherent(N, inf):
    """Returns the amplitude $\\eps = |\\alpha|$ of the coherent states giving the desired fidelity  
    to the N photon number state in the approximation - Eq? In M&A.
    """
    return (factorial(2*N+1)/(factorial(N)) * inf)**(1/(2*(N+1)))


def gen_fock_coherent(N, infid, eps = None, fast = False):
    """Generate the Bosonic state data for a Fock state N in the coherent state representation.
    
    Args:
        N (int): fock number
        infid (float): infidelity of approximation
        eps (float): coherent state amplitude, takes presidence over infid if specified.
        fast (bool): whether to invoke the fast representation, which uses approx half of the Gaussians

    See Eq 28 of http://arxiv.org/abs/2305.17099 for expansion into coherent states.

    Returns: 
        means, covs, weights, +(k if fast == True)
    """
    
    cov = 0.5*hbar * np.eye(2)
    means = []
    theta = 2*np.pi/(N+1)
    weights = []
    if not eps:
        eps = eps_fock_coherent(N, infid)

    if fast: 
        weights_re = []
        means_re = []
    
    for k in np.arange(N+1):

        alpha = eps * np.exp(1j * theta * k) 

        if fast:
            muk, cov, ck = outer_coherent(alpha, alpha)
        
            means.append(muk)
            weights.append(ck)

        for l in np.arange(N+1):
            if fast:
                if l>k: 
                    beta = eps * np.exp(1j * theta * l)
                    mukl, cov, ckl = outer_coherent(alpha, beta)
                    ckl *= np.exp(-theta * 1j* N*(k-l))
                    means_re.append(mukl)
                    weights_re.append(2*ckl)
            else:
                beta = eps * np.exp(1j * theta * l)
                mukl, cov, ckl = outer_coherent(alpha, beta)
                ckl *= np.exp(-theta * 1j* N*(k-l))
                means.append(mukl)
                weights.append(ckl)
                
    if N == 0:
        means = []
        means.append(np.array([0,0]))
        
    if fast:
        k = len(weights)
        weights = np.concatenate([weights, weights_re], axis = 0)
        means = np.concatenate([means, means_re], axis = 0)

    else:
        weights = np.array(weights)
        means = np.array(means)
    
    factor = factorial(N)/(N+1)**2 * np.exp(eps**2)/eps**(2*N)

    weights *= factor

    if fast:
        weights /= np.sum(weights.real) #renormalize
        return means, cov, weights, k
    else:
        weights /= np.sum(weights)
        return means, cov, weights


def eps_superpos_coherent(N, inf):
    """Returns the magnitude of the coherent states giving for the desired 
    infidelity of the Fock superposition up to photon number N.
    """
    return (factorial(N+1)*inf)**(1/(2*(N+1)))

def gen_fock_superpos_coherent(coeffs, infid, eps = None, fast = False):
    """Returns the weights, means and covariance matrix of the state |psi> = c0 |0> + c1 |1> + c2 |2> + ... + c_max |n_max>
    in the coherent-fock representation.

    Args:
        coeff (list/array):  the coefficients in front of the number states, coeff = [c0, c1, c2, ... c_nmax] 
        infid (float): infidelity of approx
    Returns: 
        means_new (ndarray): list of means
        cov (array): vacuum cov
        weights_new (ndarray): list of weights 
    """
    def get_ck(k, N, coeffs, eps):
        """Get coefficient ck in Eq (22) in http://arxiv.org/abs/2305.17099 for a superposition of Fock states
    
        Args:
            k (int) : coefficient number
            N (int) : max Fock number in superposition
            coeffs (list/array) 
            eps (float) 
        Returns:
            ck (complex) : k'th coefficient
        """
        ck = 0
        for n, an in enumerate(coeffs):
            ck += np.exp(eps**2 / 2)/(N+1) * np.sqrt(factorial(n)*1.0) / eps**n * an * np.exp(-2*np.pi * 1j * n * k /(N+1))
            
        return ck
    
    weights = []
    means = []

    if fast: 
        weights_re = []
        means_re = []

    N = len(coeffs)-1
    
    ck = np.zeros(N+1, dtype = 'complex128')
    if not eps:
        eps = eps_superpos_coherent(N, infid)
    
    #Obtain new coefficients
    for i in np.arange(N+1):
        ck[i] = get_ck(i, N, coeffs, eps)
        
    theta = 2*np.pi /(N+1) 

    for i, cn in enumerate(ck):
        alpha = eps * np.exp(1j * theta * i)
        if fast: 
            mui, cov, ci = outer_coherent(alpha, alpha)
            means.append(mui)
            weights.append(np.abs(cn)**2 *ci)
        
        for j, cm in enumerate(ck):
            if fast:
                if j > i:
                    cm = cm.conjugate()
                    beta = eps * np.exp(1j * theta * j) 
                    muij, cov, cij = outer_coherent(alpha, beta)
        
                    weights_re.append(2*cn*cm*cij)
                    means_re.append(muij)
            else:
                cm = cm.conjugate()
                beta = eps * np.exp(1j * theta * j) 
                muij, cov, cij = outer_coherent(alpha, beta)
    
                weights.append(cn*cm*cij)
                means.append(muij)
    if fast:
        k = len(weights)
        weights = np.concatenate([weights, weights_re], axis = 0)
        means = np.concatenate([means, means_re], axis = 0)
        weights /= np.sum(weights.real)
        return means, cov, weights, k
    else:
        weights /=np.sum(weights)
        return np.array(means), cov, np.array(weights)

def norm_coherent(N, eps):
    """REVISE
    """
    norm = 1 + eps**(2*(N+1))/factorial(N+1)
    return norm
    
def order_infidelity_fock_coherent(N, alpha):
    """give infidelity of N fock approximation using given alpha - Eq? of M&A
    """
    return factorial(N)/factorial(2*N+1)*alpha**(2*(N+1))


# |N><M| operator
# ---------------------------------------

def fock_outer_coherent(N, M, eps1, eps2):
    """Return |N><M| operator in bosonic representation
    WORK IN PROGRESS
    OBS: Purity error
    
    Obtain the weights, means and covaraince matrix in the coherent-fock representation of the |N><M| state
    
    for M > N, calculate |M><N| and take complex conjugate 

    Args: 
        N (int)
        M (int)
        eps1 (float): Displacement of |N> approx
        eps2 (float): Displacement of |M> approx

    Returns: 
        weights_new (ndarray): list of weights
        means_new (ndarray): list of means
        cov (array): vacuum cov
    
    """
    comp = 0
    if M > N:
        #Compute |M><N| and take complex conjugate at the end
        N, M = M, N
        eps1, eps2 = eps2, eps1
        comp = 1
        
    cov = hbar /2 * np.eye(2)
    means = []
    theta_N = 2*np.pi/(N+1)
    theta_M = 2*np.pi/(M+1)
    weights = []

    
    for k in np.arange(N+1):
        Re_k = eps1*np.cos(theta_N * k)
        Im_k = eps1*np.sin(theta_N * k)


        for l in np.arange(M+1):
            
            Re_l = eps2*np.cos(theta_M * l)
            Im_l = eps2*np.sin(theta_M * l)
            
            mulk = np.sqrt(hbar/2) * np.array([Re_k + Re_l + 1j*(Im_k - Im_l), Im_k + Im_l + 1j*(Re_l - Re_k)])
            means.append(mulk)
            

            clk = np.exp(-theta_N * 1j* N*k)*np.exp(theta_M *1j* l* M) * np.exp(-0.5* (Im_k - Im_l)**2 - 0.5 *(Re_k - Re_l)**2
                                                     -1j*Im_l*Re_k + 1j*Im_k*Re_l )
            weights.append(clk)
            
    weights = np.array(weights)

    #Here do a small simplifcation of the factorial in order to be able to compute the sqrt
    K = N - M
    if K != N:
        factorial_simplify = np.sqrt(factorial(N)/factorial(N-K-1))*factorial(M)
    else:
        factorial_simplify = np.sqrt(factorial(N)*factorial(M)/1.0)
        
    #factor = np.sqrt(factorial(N)*factorial(M))/((N+1)*(M+1)) * np.exp(eps1**2/2+eps2**2/2)/(eps1**N * eps2**M)
    
    factor = factorial_simplify/((N+1)*(M+1)) * np.exp(eps1**2/2+eps2**2/2)/(eps1**N * eps2**M)

    weights *= factor
    means = np.array(means)
    weights = np.array(weights)
    
    if comp:
        means = means.conjugate()
        weights = weights.conjugate()

    ##Quick and dirty solution to fix purity issue - Generate norm of |N> and |M> and divide by sqrts of norms
    norm1 = norm_coherent(N, eps1)
    norm2 = norm_coherent(M, eps2)
    #print(f"{N,M}", 1-1/norm1, 1-1/norm2, eps1, eps2)

    weights /= np.sum(weights) 

    
    
    return means, cov, weights

def outer_sqz_coherent(r, alpha, beta, MP = False):
    """ Returns the coefficient, displacement vector and covariance matrix (vacuum) of the Gaussian that
    describes the Wigner function of the outer product of two coherent states |alpha><beta| derived 
    in Appendix A of https://arxiv.org/abs/2103.05530.
    r>0: Squeezing in x
    r<0: Squeezing in p
    """
    cov = hbar /2 * np.array([[np.exp(-2*r),0],[0,np.exp(2*r)]])
    gamma = alpha/(np.cosh(r)+np.sinh(r))
    delta = beta/(np.cosh(r)+np.sinh(r))
    
    re_gamma = gamma.real
    im_gamma = gamma.imag
    re_delta = delta.real
    im_delta = delta.imag
    
    mu = np.sqrt(hbar/2) * np.array([re_gamma + re_delta
                                        + 1j *np.exp(-2*r)*(im_gamma - im_delta), 
                                        im_gamma + im_delta
                                        + 1j * np.exp(2*r)* (re_delta - re_gamma)])

    
    if MP:
        coeff = mp.exp( -0.5 * mp.exp(-2*r)* (im_gamma - im_delta) **2
                       - 0.5 * mp.exp(2*r)*(re_gamma - re_delta) **2
                       - 1j * im_delta * re_gamma + 1j * im_gamma * re_delta)
    else:

        coeff = np.exp( -0.5 * np.exp(-2*r)* (im_gamma - im_delta) **2
                       - 0.5 * np.exp(2*r)*(re_gamma - re_delta) **2
                       - 1j * im_delta * re_gamma + 1j * im_gamma * re_delta)

    return mu, cov, coeff

def gen_sqz_cat_coherent(r, alpha, k, MP = False):
    """Prepare a squeezed cat, requires a higher precision with mp.math

    Args: 
        r : squeezing of the cat
        alpha: displacement of the cat (pre-squeezing)
        k : parity
    Returns:
        tuple
    """
    
    params = [(1, alpha,alpha), (1,-alpha,-alpha), ((-1)**k,alpha,-alpha), ((-1)**k,-alpha,alpha)]
    means = []
    weights = []
    
    for a in params:
        means_a, cov, weights_a = outer_sqz_coherent(r, a[1], a[2],MP)
        means.append(means_a)
        weights.append(weights_a*a[0])
    if MP: 
        weights = weights/ np.array(mp.fsum(weights))
    else:
        weights = weights/ np.array(np.sum(weights))
    
    return np.array(means), cov, weights


def gen_fock_bosonic(n, r=0.05):
    """
    Prepares the arrays of weights, means and covs of a Fock state.
    Normalisation becomes zero for n > 6 giving nan in the weights

    Copied from strawberryfields bosonicbackend, modified here.

    Args:
        n (int): photon number
        r (float): quality parameter for the approximation

    Returns:
        fock (BaseBosonicState): Fock state object

    Raises:
        ValueError: if :math:`1/r^2` is less than :math:`n`
    """
    if 1 / r**2 < n:
        raise ValueError(f"The parameter 1 / r ** 2={1 / r ** 2} is smaller than n={n}")
    # A simple function to calculate the parity
    parity = lambda n: 1 if n % 2 == 0 else -1
    # All the means are zero
    means = np.zeros([n + 1, 2])
    covs = np.array(
        [
            #0.5
            1
            #* sf.hbar
            * np.identity(2)
            * (1 + (n - j) * r**2)
            / (1 - (n - j) * r**2)
            for j in range(n + 1)
        ]
    )
    weights = np.array(
        [
            (1 - n * (r**2)) / (1 - (n - j) * (r**2)) * comb(n, j) * parity(j)
            for j in range(n + 1)
        ],
    )
    #weights /= np.sum(weights)


    return means, covs, weights


