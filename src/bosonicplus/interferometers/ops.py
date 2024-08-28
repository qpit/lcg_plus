import numpy as np
import thewalrus.symplectic as symp

def apply_symplectic(data, S):
    r""" Apply symplectic to data in coherent basis.
    data (tuple) : [means, covs, weights]
    S (ndarray) : symplectic matrix
    """
    means, covs, weights = data
    #check that S has the correct dimensions
    if np.shape(S)[0] != int(np.shape(means)[-1]):
        raise ValueError('S must must be 2nmodes x 2nmodes. ')

    
    new_means = np.einsum("...jk,...k", S[np.newaxis,:], means)
    new_covs = S @ covs @ S.T
    return new_means, new_covs, weights

def apply_displacement(data, disp):
    r""" disp in xpxp notation
    """
    means, covs, weights = data
    return means+disp, covs, weights


def apply_loss(data, etas, nbars):
    """Apply loss to (multimode) state in data

    Gaussian state undergo a loss/thermal loss channel in the following way:
        cov = X @ cov @ X.T + Y
        means = X @ means

    Args:
        data ([means, cov, weights]): input state
        etas (array): array giving transmittivity of each mode in data
        nbars (array): array giving number of photons in environment each mode is coupled to

    Returns:
        data_loss: state after loss
    """

    means, cov, weights = data
    num_modes = int(cov.shape[-1]/2)

    #First multiply cov with diag(etas)
    X = symp.xxpp_to_xpxp(np.diag(np.repeat(np.sqrt(etas),2)))
    cov = X @ cov @ X.T
    
    #Multiply means with diag(etas)
    means = np.einsum("...jk,...k", X, means)
    
    #Make Y and add to cov
    Y = symp.xxpp_to_xpxp(np.diag(np.repeat( (1-etas)*(sf.hbar/2) * (2*nbars + 1) ,2 )))
    
    cov += Y
    
    data_loss = means, cov, weights
    #state_loss = BaseBosonicState(data_loss, num_modes = num_modes, num_weights = len(weights))

    return data_loss
    