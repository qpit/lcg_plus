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
from lcg_plus.charfun import char_fun, char_fun_gradients

def get_gkp_stabilizer_displacement(lattice):

    lattices = ['sx', 'sp', 'rx', 'rp', 'hx', 'hp', 'hsx', 'hsp']
        
    if lattice not in lattices:
        raise ValueError('Lattice must be either sx, sp, rx, rp, hx, hp, hsx or hsp.')
        
    kappa_p = np.sqrt(np.pi/8)*(3**(1/4) + 3**(-1/4))
    kappa_m = np.sqrt(np.pi/8)*(3**(1/4) - 3**(-1/4))

    kappa1 = 3**(-1/4) + 3**(1/4)
    kappa2 = 3**(-1/4) - 3**(1/4)

    #Terhal definition
    if lattice == 'sx':
        alpha = 1j*np.sqrt(np.pi) 
        
    elif lattice == 'sp':        
        alpha = np.sqrt(np.pi)
        
    elif lattice == 'rx':
        alpha = 1j*np.sqrt(2*np.pi)
        
    elif lattice == 'rp':
        #alpha = np.sqrt(np.pi/2)
        alpha = np.sqrt(2*np.pi)
        
    elif lattice == 'hx':
        alpha = np.sqrt(np.pi/2) * (kappa1+ 1j*kappa2)
    elif lattice == 'hp':
        alpha = np.sqrt(np.pi/2) * (kappa2+ 1j*kappa1)
    elif lattice == 'hsx':
        alpha = np.sqrt(np.pi)/2 * (kappa1 + 1j*kappa2)
    elif lattice == 'hsp': 
        alpha = np.sqrt(np.pi)/2 * (kappa2 + 1j*kappa1)
        
    return alpha
    

def effective_squeezing(state, lattice : str):
    """Get the effective squeezing of a state according to the Terhal definition.

    Args:
        state : base.State object
        lattice: The GKP lattice and direction, sx, sp, rx, rp, hx, hp, hsx, hsp
    """

    alpha = get_gkp_stabilizer_displacement(lattice)

    
    #Symmetrically
    f1 = char_fun(state, alpha)
    f2 = char_fun(state, -alpha)

    
    D1 = np.sqrt(-2/np.abs(alpha)**2*np.log(np.abs(f1)))
    D2 = np.sqrt(-2/np.abs(alpha)**2*np.log(np.abs(f2)))
    
    Delta = 0.5 * (D1+D2)
    
    return float(Delta)


def effective_squeezing_gradients(state, lattice : str):
    """
    Args:
        state : State
        lattice: The GKP lattice and direction, sx, sp, rx, rp, hx, hp, hsx, hsp
    """

    alpha = get_gkp_stabilizer_displacement(lattice)

    f1, df1 = char_fun_gradients(state, alpha)
    f2, df2 = char_fun_gradients(state, -alpha)

    Delta_plus_squared = -2/np.abs(alpha)**2*np.log(np.abs(f1)) #Delta(+alpha)^2
    Delta_minus_squared = -2/np.abs(alpha)**2*np.log(np.abs(f2)) #Delta(-alpha)^2


    grad_Delta_plus_squared  = - 2/np.abs(alpha)**2 * f1 / np.abs(f1)**2 * df1 #gradient of Delta(+alpha)^2
    grad_Delta_minus_squared = - 2/np.abs(alpha)**2 * f2 / np.abs(f2)**2 * df2 #gradient of Delta(-alpha)^2
    
    Delta =  0.5 * ( np.sqrt(Delta_plus_squared) + np.sqrt(Delta_minus_squared) )
    dDelta =  0.25 * (grad_Delta_plus_squared / np.sqrt(Delta_plus_squared) + grad_Delta_minus_squared / np.sqrt(Delta_minus_squared))
    
    return float(Delta), dDelta

def effective_squeezing_squared_gradients(state, lattice : str):
    """
    Args:
        state : State
        lattice: The GKP lattice and direction, sx, sp, rx, rp, hx, hp, hsx, hsp
    """

    alpha = get_gkp_stabilizer_displacement(lattice)

    f1, df1 = char_fun_gradients(state, alpha) #Xi(+alpha)
    f2, df2 = char_fun_gradients(state, -alpha) #Xi(-alpha)

    Delta_plus_squared = -2/np.abs(alpha)**2*np.log(np.abs(f1)) #Delta(+alpha)^2
    Delta_minus_squared = -2/np.abs(alpha)**2*np.log(np.abs(f2)) #Delta(-alpha)^2

    grad_Delta_plus_squared =  - 2/np.abs(alpha)**2 * f1 / np.abs(f1)**2 * df1 #gradient of Delta(+alpha)^2
    grad_Delta_minus_squared = - 2/np.abs(alpha)**2 * f2 / np.abs(f2)**2 * df2 #gradient of Delta(-alpha)^2
    
    Delta_squared =  0.5 * (Delta_plus_squared + Delta_minus_squared)
    grad_Delta_squared =  0.5 * (grad_Delta_plus_squared + grad_Delta_minus_squared)
    
    return float(Delta_squared), grad_Delta_squared





