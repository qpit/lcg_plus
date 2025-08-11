# Copyright © 2025 Technical University of Denmark

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

def r_to_dB(r):
    """Convert squeezing to dB units, r_dB = 10 * log10( exp(-2r) )
    """
    return 10*np.log10(np.exp(-2*r))

def eps_to_dB(eps):
    """Convert fock damping to dB units (Hastrup loss analysis)
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

def to_dB(x):
    return -10*np.log10(x)
