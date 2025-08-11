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


from lcg_plus.operations.circuit_parameters import gen_interferometer_params, params_to_1D_array, unpack_params, params_to_dict
from lcg_plus.operations.gbs import build_interferometer, build_interferometer_gradients
from lcg_plus.conversions import dB_to_r, r_to_dB, Delta_to_dB
import numpy as np
from lcg_plus.effective_sqz import effective_sqz, effective_sqz_gradients
from lcg_plus.gkp_squeezing import Q_expval, Q_expval_gradients

def state_prep_GBS(params, 
                   nmodes, 
                   pattern, 
                   bs_arrange='Clements', 
                   setting = 'no_phase',
                   eta=None, 
                   nbar=None, 
                   pPNR = None,
                   gradients = False,
                   inf = 1e-4, 
                   fast= False):
    
    """ Compute output state and the measurement probability of an GBS with nmodes.

    Args:    
        params (ndarry): is a 1D array to be used as the argument in the costfunction, which must be converted into correct params dict for build_interferometer function.
        nmodes (int): number of modes
        pattern (list): photon number pattern (pattern.len = nmodes - 1) 
        bs_arrange (str): Clements, cascade, inv_cascade, the entangling gate (beamsplitter) arrangement
        setting (str): no_phase or two_mode_squeezing
        eta (ndarray or None): sqrt(transmissivity) before PNRD (default is no loss)
        nbar (ndarray or None): dark counts (default i none)
        pPNRD (int or None) : if pseudo-PNRD is selected, the number of on/off detectors per pPNRD
        gradients (bool): optimise with gradients or not
        inf (float): Infidelity of coherent state approx of PNR |n><n| POVM 
    Returns: 
        state_out (BaseBosonicState), the measurement probability is the state.norm    
    """

    params_dict = params_to_dict(params, nmodes, bs_arrange, setting) #convert 1D array into a dictionary
    
    if gradients:
        state = build_interferometer_gradients(params_dict, nmodes, setting=setting)
    else:

        state = build_interferometer(params_dict, nmodes, setting=setting)

    # Apply loss 
    
    if isinstance(eta, np.ndarray):
        #check if eta and nbar have the right shape
        if len(eta) != nmodes:
            raise ValueError('eta must be an array with size nmodes.')
        if isinstance(nbar, np.ndarray):
            if len(nbar) != nmodes: 
                raise ValueError('nbar must be an array with size nmodes.')
        else: nbar = np.zeros(nmodes) #pure loss
       
        state.apply_loss(eta, nbar)
  
    for i in range(nmodes-1):
    
        if pPNR:
            state.post_select_ppnrd_thermal(0, pattern[i], pPNR) 
            if gradients:
                raise ValueError('gradients not implemented for ppnr.')
        
        else:
            if gradients:
                state.post_select_fock_coherent_gradients(0, pattern[i], inf) #Will always be fast=False
            else:
                state.post_select_fock_coherent(0, pattern[i], inf, red_gauss = fast) 
            
        
    return state


def symm_effective_squeezing(*args):
    """
    """
    state = state_prep_GBS(*args[0:-2])
    lattice = args[-1]
    
    eff = effective_sqz(state, lattice+'x') + effective_sqz(state, lattice+'p')
        
    return eff.real/2

def gkp_squeezing(*args):
    """
    Computes the expectation value of the GKP non-linear squeezing operator for a state
    prepared with GBS
    """
    
    state = state_prep_GBS(*args[0:-2])
    lattice = args[-1]
    expval = Q_expval(state, lattice)
        
    return expval.real 


def symm_effective_squeezing_gradients(*args):
    """
    """
    
    state = state_prep_GBS(*args[0:-2])
    lattice = args[-1]
    Dx, Dx_grad = effective_sqz_gradients(state, lattice+'x')
    Dp, Dp_grad = effective_sqz_gradients(state, lattice+'p')

    eff = Dx + Dp  #Should be np.sqrt(0.5*(Dx**2 + Dp**2))
    df = Dx_grad + Dp_grad
        
    return 0.5*eff.real, 0.5*df.real

def gkp_squeezing_gradients(*args):
    """
    """
    
    state = state_prep_GBS(*args[0:-2])
    lattice = args[-1]
    expval, dexpval = Q_expval_gradients(state, lattice)
        
    return expval.real, dexpval.real


