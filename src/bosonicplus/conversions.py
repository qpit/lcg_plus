#Functions for converting to and from dB units

import numpy as np
#import mpmath

def r_to_dB(r):
    """Convert squeezing to dB units
    """
    return 10*np.log10(np.exp(-2*r))

def eps_to_dB(eps):
    """Convert fock damping to dB units
    """
    return -10*np.log10(np.tanh(eps))
    
def dB_to_r(dB):
    """Convert from dB squeezing to squeezing value
    """
    return -0.5*np.log(10**(dB/10))

#def Delta_to_dB_MP(Delta):
 #   return float(-20*mpmath.log(Delta,10))

def Delta_to_dB(Delta):
    return -10*np.log10(Delta**2)

def dB_to_Delta(Delta_dB):
    return 10**(-Delta_dB/20)


#def ratio_to_dB(amp):
 #   """ Convert amplitude ratio to dB
  #  """
   # return -20*np.log10(amp)

#def dB_to_ratio(dB):
 #   """ Convert dB to amplitude ratio
  #  """
   # return 10**(-dB/20)

#def Delta_to_epsilon(Delta):
 #   """ Convert Delta to epsilon squeezing
  #  """
   # return Delta **2

#def epsilon_to_Delta(epsilon):
 #   """ Convert epsilon to Delta squeezing
  #  """
   # return np.sqrt(epsilon)

#def epsilon_to_dB(epsilon):
 #   """ Convert epsilon to dB squeezing
  #  """
   # Delta = epsilon_to_Delta(epsilon)
    #return ratio_to_dB(Delta)

#def Delta_dB_to_epsilon(dB):
 #   """Convert dB to epsilon
  #  """
   # return dB_to_power(-dB)

