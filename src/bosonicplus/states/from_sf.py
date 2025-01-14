import numpy as np
import itertools as it

from mpmath import mp
from scipy.special import comb
from math import fsum
from bosonicplus.base import State

hbar = 2

def prepare_gkp_bosonic(state, epsilon, ampl_cutoff = 1e-12, representation="real", shape="square"):
        r"""
        Copied from strawberryfields bosonicbackend, modified here.
        
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

        means *= 0.5 * damping * np.sqrt(np.pi * hbar)
        # Covariances all the same
        covs = (
            0.5
            * hbar
            * (1 - np.exp(-2 * epsilon))
            / (1 + np.exp(-2 * epsilon))
            * np.identity(2)
        )
        covs = np.repeat(covs[None, :], weights.size, axis=0)
        
        state = State(1)
        state.update_data([means, covs, weights])
        return state


def prepare_fock_bosonic(n, r=0.05):
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

    state = State(1)
    state.update_data([means, covs, weights])
    state.norm = np.sum(weights)

    return state

def prepare_cat_bosonic(a, theta, p, MP = False):
    r"""Prepares the arrays of weights, means and covs for a cat state:

    Copied from strawberryfields bosonicbackend, modified.  
    
    :math:`\ket{\text{cat}(\alpha)} = \frac{1}{N} (\ket{\alpha} +e^{i\phi} \ket{-\alpha})`,
    
    where :math:`\alpha = ae^{i\theta}`.
    
    Args:
        a (float): displacement magnitude :math:`|\alpha|`
        theta (float): displacement angle :math:`\theta`
        p (float): Parity, where :math:`\phi=p\pi`. ``p=0`` corresponds to an even
            cat state, and ``p=1`` an odd cat state.
        MP (bool): Use mpmath for complex coefficients or not
        
    
    Returns:
        cat (BaseBosonicState): Cat state object
    """
    
    phi = np.pi * p
    # Case alpha = 0, prepare vacuum
    if np.isclose(a, 0):
        weights = np.array([1], dtype=complex)
        means = np.array([[0, 0]], dtype=complex)
        covs = np.array([0.5 * hbar * np.identity(2)])
        state = State(1)
        state.update_data([means, covs, weights])
        return state
    
    # Normalization factor
    norm = 1 / (2 * (1 + np.exp(-2 * a**2) * np.cos(phi)))
    
    alpha = a * np.exp(1j * theta)
    
    # Mean of |alpha><alpha| term
    rplus = np.sqrt(2 * hbar) * np.array([alpha.real, alpha.imag])
    
    # Mean of |alpha><-alpha| term
    rcomplex = np.sqrt(2 * hbar) * np.array([1j * alpha.imag, -1j * alpha.real])
    
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

    state = State(1)
    state.update_data([means, covs, weights])
    
    return state
