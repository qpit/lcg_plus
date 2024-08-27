import numpy as np
from math import factorial
import strawberryfields as sf
from strawberryfields.backends.states import BaseBosonicState
from strawberryfields.backends.bosonicbackend.bosoniccircuit import BosonicModes
import itertools as it
from scipy.special import comb, genlaguerre
from scipy.linalg import block_diag
from mpmath import mp


# Represent non-Gaussian states in the sf.bosonicbackend format.

def prepare_gkp_bosonic(state, epsilon, ampl_cutoff = 1e-12, representation="real", shape="square"):
        r"""
        Copied from strawberryfields bosonicbackend.
        
        Prepares the arrays of weights, means and covs for a finite energy GKP state.
        GKP states are qubits, with the qubit state defined by:
        :math:`\ket{\psi}_{gkp} = \cos\frac{\theta}{2}\ket{0}_{gkp} + e^{-i\phi}\sin\frac{\theta}{2}\ket{1}_{gkp}`
        where the computational basis states are :math:`\ket{\mu}_{gkp} = \sum_{n} \ket{(2n+\mu)\sqrt{\pi\hbar}}_{q}`.
        Args:
            state (list): ``[theta,phi]`` for qubit definition above
            epsilon (float): finite energy parameter of the state
            ampl_cutoff (float): this determines how many terms to keep
            representation (str): ``'real'`` or ``'complex'`` reprsentation
            shape (str): shape of the lattice; default 'square'
            
        Returns:
            gkp (BaseBosonicState): gkp state object
            
        Raises:
            NotImplementedError: if the complex representation or a non-square lattice is attempted
        """

        if representation == "complex":
            raise NotImplementedError("The complex description of GKP is not implemented")

        if shape != "square":
            raise NotImplementedError("Only square GKP are implemented for now")

        theta, phi = state[0], state[1]

        def coeff(peak_loc):
            """Returns the value of the weight for a given peak.
            Args:
                peak_loc (array): location of the ideal peak in phase space
            Returns:
                float: weight of the peak
            """
            l, m = peak_loc[:, 0], peak_loc[:, 1]
            t = np.zeros(peak_loc.shape[0], dtype=complex)
            t += np.logical_and(l % 2 == 0, m % 2 == 0)
            t += np.logical_and(l % 4 == 0, m % 2 == 1) * (
                np.cos(0.5 * theta) ** 2 - np.sin(0.5 * theta) ** 2
            )
            t += np.logical_and(l % 4 == 2, m % 2 == 1) * (
                np.sin(0.5 * theta) ** 2 - np.cos(0.5 * theta) ** 2
            )
            t += np.logical_and(l % 4 % 2 == 1, m % 4 == 0) * np.sin(theta) * np.cos(phi)
            t -= np.logical_and(l % 4 % 2 == 1, m % 4 == 2) * np.sin(theta) * np.cos(phi)
            t -= (
                np.logical_or(
                    np.logical_and(l % 4 == 3, m % 4 == 3),
                    np.logical_and(l % 4 == 1, m % 4 == 1),
                )
                * np.sin(theta)
                * np.sin(phi)
            )
            t += (
                np.logical_or(
                    np.logical_and(l % 4 == 3, m % 4 == 1),
                    np.logical_and(l % 4 == 1, m % 4 == 3),
                )
                * np.sin(theta)
                * np.sin(phi)
            )
            prefactor = np.exp(
                -np.pi
                * 0.25
                * (l**2 + m**2)
                * (1 - np.exp(-2 * epsilon))
                / (1 + np.exp(-2 * epsilon))
            )
            weight = t * prefactor
            return weight

        # Set the max peak value
        z_max = int(
            np.ceil(
                np.sqrt(
                    -4
                    / np.pi
                    * np.log(ampl_cutoff)
                    * (1 + np.exp(-2 * epsilon))
                    / (1 - np.exp(-2 * epsilon))
                )
            )
        )
        damping = 2 * np.exp(-epsilon) / (1 + np.exp(-2 * epsilon))

        # Create set of means before finite energy effects
        means_gen = it.tee(
            it.starmap(lambda l, m: l + 1j * m, it.product(range(-z_max, z_max + 1), repeat=2)),
            2,
        )
        means = np.concatenate(
            (
                np.reshape(
                    np.fromiter(means_gen[0], complex, count=(2 * z_max + 1) ** 2), (-1, 1)
                ).real,
                np.reshape(
                    np.fromiter(means_gen[1], complex, count=(2 * z_max + 1) ** 2), (-1, 1)
                ).imag,
            ),
            axis=1,
        )

        # Calculate the weights for each peak
        weights = coeff(means)
        filt = abs(weights) > ampl_cutoff
        weights = weights[filt]

        weights /= np.sum(weights)
        # Apply finite energy effect to means
        means = means[filt]

        means *= 0.5 * damping * np.sqrt(np.pi * sf.hbar)
        # Covariances all the same
        covs = (
            0.5
            * sf.hbar
            * (1 - np.exp(-2 * epsilon))
            / (1 + np.exp(-2 * epsilon))
            * np.identity(2)
        )
        covs = np.repeat(covs[None, :], weights.size, axis=0)
        
        gkp = BaseBosonicState([means, covs, weights], num_modes = 1, num_weights = len(weights))
        
        return gkp
    
def prepare_sqz_state(r):
    """
    Prepare state squeezed in x by r.
    (To do: generalise to squeezing in an arbitrary direction)
    
    Args:
        r (float): squeezing parameter
    Returns:
        state_sq (BaseBosonicState): squeezed state object
    """ 
    covmat = sf.hbar / 2 * np.array([[np.exp(-2*r), 0], [0,np.exp(2*r)]])
    mus = np.zeros(2)
    

    state_sq = BaseBosonicState([ mus[np.newaxis,:], covmat[np.newaxis,:], np.array([1])], 
                                num_modes = 1, num_weights = 1)
    return state_sq
    
    
def prepare_fock_bosonic(n, r=0.05):
    """
    Prepares the arrays of weights, means and covs of a Fock state. 
    Normalisation becomes zero for n > 6 giving nan in the weights

    Copied from strawberryfields bosonicbackend. 

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
        dtype=complex,
    )
    weights /= np.sum(weights)

    fock = BaseBosonicState([means, covs, weights], num_modes = 1, num_weights = len(weights))

    return fock

def prepare_cat_bosonic(a, theta, p, MP = False):
    r"""Prepares the arrays of weights, means and covs for a cat state:
    
    :math:`\ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{i\phi} \ket{-\alpha})`,
    
    where :math:`\alpha = ae^{i\theta}`.
    
    Args:
        a (float): displacement magnitude :math:`|\alpha|`
        theta (float): displacement angle :math:`\theta`
        p (float): Parity, where :math:`\phi=p\pi`. ``p=0`` corresponds to an even
            cat state, and ``p=1`` an odd cat state.
        representation (str): whether to use the ``'real'`` or ``'complex'`` representation
        ampl_cutoff (float): if using the ``'real'`` representation, this determines
             how many terms to keep
        D (float): for ``'real'`` representation, quality parameter of approximation
    
    Returns:
        tuple: arrays of the weights, means and covariances for the state
    """
    
    phi = np.pi * p
    # Case alpha = 0, prepare vacuum
    if np.isclose(a, 0):
        weights = np.array([1], dtype=complex)
        means = np.array([[0, 0]], dtype=complex)
        covs = np.array([0.5 * sf.hbar * np.identity(2)])
        return BaseBosonicState([means, covs, weights], num_modes = 1, num_weights = 1)
    
    # Normalization factor
    norm = 1 / (2 * (1 + np.exp(-2 * a**2) * np.cos(phi)))
    hbar = sf.hbar
    
    alpha = a * np.exp(1j * theta)
    
    # Mean of |alpha><alpha| term
    rplus = np.sqrt(2 * sf.hbar) * np.array([alpha.real, alpha.imag])
    
    # Mean of |alpha><-alpha| term
    rcomplex = np.sqrt(2 * sf.hbar) * np.array([1j * alpha.imag, -1j * alpha.real])
    
     # Coefficient for complex Gaussians
    if MP:
        cplx_coef = mp.exp(-2*np.absolute(alpha)**2 -1j*phi)
    else:
       
        cplx_coef = np.exp(-2 * np.absolute(alpha) ** 2 - 1j * phi)
    
    # Arrays of weights, means and covs
    weights = norm * np.array([1, 1, cplx_coef, np.conjugate(cplx_coef)])
    weights /= np.sum(weights)
    
    means = np.array([rplus, -rplus, rcomplex, np.conjugate(rcomplex)])
    
    covs = 0.5 * hbar * np.identity(2, dtype=float)
    #covs = np.repeat(covs[None, :], weights.size, axis=0)
    cat = BaseBosonicState([means, covs, weights], num_modes = 1, num_weights=4)
    
    return cat


# COHERENT STATE SIM TOOLS 
# ----------------------------------
# Simulation tools for approximating Fock state superpositions as
# superpositions of coherent states from Marshall & Anand http://arxiv.org/abs/2305.17099.

# The Wigner function of the superposition of coherent states is a weighted sum of Gaussians.


def outer_coherent(alpha, beta):
    """ Returns the coefficient, displacement vector and covariance matrix (vacuum) of the Gaussian that
    describes the Wigner function of the outer product of two coherent states |alpha><beta| derived 
    in Appendix A of https://arxiv.org/abs/2103.05530.
    """
    cov = sf.hbar /2 * np.eye(2)
    re_alpha = alpha.real
    im_alpha = alpha.imag
    re_beta = beta.real
    im_beta = beta.imag

    mu = np.sqrt(sf.hbar/2) * np.array([re_alpha + re_beta 
                                        + 1j *(im_alpha - im_beta), 
                                        im_alpha + im_beta
                                        + 1j * (re_beta - re_alpha)])
    coeff = np.exp( -0.5 * (im_alpha - im_beta) **2
                   - 0.5 * (re_alpha - re_beta) **2
                   - 1j * im_beta * re_alpha + 1j * im_alpha * re_beta)

    return mu, cov, coeff

def eps_fock_coherent(N, inf):
    """Returns the magnitude of the coherent states giving for the desired infidelity of 
    the N Fock approximation.
    """
    return (factorial(2*N+1)/(factorial(N)) * inf)**(1/(2*(N+1)))


def gen_fock_coherent(N, infid, eps = None):
    """Generate the Gaussian state data for a Fock state N in the coherent state representation.
    
    Args:
        N (int): fock number
        infid (float): infidelity of approximation
        
    See Eq 28 of http://arxiv.org/abs/2305.17099 for expansion into coherent states.
    """
    
    
    cov = 0.5*sf.hbar * np.eye(2)
    means = []
    theta = 2*np.pi/(N+1)
    weights = []
    if not eps:
        eps = eps_fock_coherent(N, infid)
    
    for k in np.arange(N+1):

        alpha = eps * np.exp(1j * theta * k) 

        for l in np.arange(N+1):
    
            beta = eps * np.exp(1j * theta * l)

            mukl, cov, ckl = outer_coherent(alpha, beta)

            ckl *= np.exp(-theta * 1j* N*(k-l))
            
            means.append(mukl)
            weights.append(ckl)

    if N == 0:
        means = []
        means.append(np.array([0,0]))
        
            
    weights = np.array(weights)
    
    factor = factorial(N)/(N+1)**2 * np.exp(eps**2)/eps**(2*N)

    weights *= factor

    weights /= np.sum(weights) #renormalize
    
    return np.array(means), cov, weights
    

def eps_superpos_coherent(N, inf):
    """Returns the magnitude of the coherent states giving for the desired 
    infidelity of the Fock superposition up to photon number N.
    """
    return (factorial(N+1)*inf)**(1/(2*(N+1)))


def gen_fock_superpos_coherent(coeffs, infid, eps = None):
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

    N = len(coeffs)-1
    
    ck = np.zeros(N+1, dtype = 'complex128')
    if not eps:
        eps = eps_superpos_coherent(N, infid)
    
    #Obtain new coefficients
    for i in np.arange(N+1):
        ck[i] = get_ck(i,N,coeffs, eps)
        
    theta = 2*np.pi /(N+1) 

    for i, cn in enumerate(ck):
        alpha = eps * np.exp(1j * theta * i)
        
        for j, cm in enumerate(ck):
           
            cm = cm.conjugate()
            beta = eps * np.exp(1j * theta * j) 

            muij, cov, cij = outer_coherent(alpha, beta)

            weights.append(cn*cm*cij)
            means.append(muij)
            
    weights /= np.sum(weights) #Renormalize
    
    return np.array(means), cov, np.array(weights)

def norm_coherent(N, eps):
    """REVISE
    """
    norm = 1 + eps**(2*(N+1))/factorial(N+1)
    return norm
    
def order_infidelity_fock_coherent(N, alpha):
    """give infidelity of N fock approximation using given alpha
    """
    return factorial(N)/factorial(2*N+1)*alpha**(2*(N+1))
# |N><M| state
# ---------------------------------------

def fock_outer_coherent(N, M, eps1, eps2):
    """
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
        
    cov = sf.hbar /2 * np.eye(2)
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
            
            mulk = np.sqrt(sf.hbar/2) * np.array([Re_k + Re_l + 1j*(Im_k - Im_l), Im_k + Im_l + 1j*(Re_l - Re_k)])
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


# MULTIMODE COPIES
# ----------------------------------------

def multimode_copy(state, num_modes):
    """Copy the state stored in data to num_modes copies.
    """
    
    # Check number of modes in state
    if state.num_modes != 1:
        raise ValueError('The state has multiple modes. Can only copy make copies of single mode states.')
        
    means, cov, weights = state.data
    
    
    new_weights = np.prod(np.array(list(it.product(weights.tolist(), repeat = num_modes, ))), axis = 1)
    new_means = np.reshape(np.array(list(it.product(means, repeat = num_modes))), (len(weights)**num_modes, num_modes*2) )
    new_cov = np.array([block_diag(*tup) for tup in list(it.product([cov], repeat = num_modes))])

    
    data_new = new_means, new_cov, new_weights
    
    multimode_state = BaseBosonicState(data_new, num_modes = num_modes, num_weights = len(new_weights))
    
    return multimode_state


# WIGNER FUNCTIONS
# ----------------------------------------

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

    #Check cov shape
    if cov.shape[0] != 1: 
        raise ValueError('cov is not in the coherent rep. Use state.wigner() instead.')
    
    W = 0
        
    for i, mu in enumerate(means):
        W += weights[i] * Gauss(cov, mu, x, p)
    return W
