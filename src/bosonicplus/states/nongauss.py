import numpy as np
from math import factorial
from bosonicplus.base import State
from .coherent import gen_fock_coherent, gen_sqz_cat_coherent
hbar = 2

def prepare_fock_coherent(n, inf=1e-4, epsilon = None):
    """Prepare Fock state in coherent state approx"""
    data = gen_fock_coherent(n, inf,epsilon)
    fock = State(1)
    fock.update_data(data)
    return fock

def prepare_sqz_cat_coherent(r, alpha, k):
    """Prepare a squeezed cat, requires a higher precision with mp.math
    Args: 
        r : squeezing of the cat
        alpha: displacement of the cat (pre-squeezing)
        k : parity
    Returns:
        State
    
    """
    data = gen_sqz_cat_coherent(r, alpha, k)
    
    sq_cat = State(1)
    sq_cat.update_data(data)
    return sq_cat
    