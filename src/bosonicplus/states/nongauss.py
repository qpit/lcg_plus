import numpy as np
from strawberryfields.backends.states import BaseBosonicState
from .coherent import gen_fock_superpos_coherent, gen_fock_coherent

from .gkp_squeezing import gkp_nonlinear_squeezing_operator

def prepare_gkp_coherent(n, type, N = 1,inf = 1e-4):
    """
    Obtain best GKP state in coherent state decomp from the ground state of the GKP nonlinear squeezing operator
    Args: 
        n: Fock cutoff
        type: '0', '1', 's0', 's1', 'h'
        N: scaling of the grid
        inf: (in)fidelity of the coherent state approximation
    """
    rho = gkp_nonlinear_squeezing_operator(n, N=N, type = type)

    w, v = np.linalg.eigh(rho)
    
    coeffs = v[:,0] #eigs always sorted from lowest to highest eigenvalue, choose lowest
    data_GKP = gen_fock_superpos_coherent(coeffs, inf)

    gkp = BaseBosonicState(data_GKP, num_modes = 1, num_weights = len(data_GKP[-1]))
    
    return gkp

def prepare_fock_coherent(n, inf=1e-4):
    """Prepare Fock state in coherent state approx"""
    data = gen_fock_coherent(n, inf)
    fock = BaseBosonicState(data, num_modes = 1, num_weights = len(data[-1]))
    return fock


