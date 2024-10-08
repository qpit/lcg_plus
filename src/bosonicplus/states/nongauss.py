import numpy as np
from math import factorial
from bosonicplus.base import State
from .coherent import gen_fock_coherent
hbar = 2

def prepare_fock_coherent(n, inf=1e-4, epsilon = None):
    
    """Prepare Fock state in coherent state approx"""
    data = gen_fock_coherent(n, inf,epsilon)
    fock = State(1)
    fock.update_data(data)
    return fock
    