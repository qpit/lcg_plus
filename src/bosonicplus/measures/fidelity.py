import numpy as np
import strawberryfields as sf
from strawberryfields.backends.bosonicbackend.bosoniccircuit import BosonicModes
from strawberryfields.backends.states import BaseBosonicState
from bosonicplus.conversions import *
from mpmath import mp


# Fidelity with coherent states superpositions. 
# The coherent states have the vacuum covariance matrix, making the fidelity calculation faster.

def fidelity_coherent(state1, state2, MP = False):
    """Calculate fidelity between two states in coherent representation 
    i.e. there is only one covariance matrix, but many weights and means.
    
    Args:
        state1 (BaseBosonicState):
        state2 (BaseBosonicState):
        
    Returns:
        fidelity (float): fidelity of state1 with state2, assuming one of them is pure
    """
    
    #Equal number of modes check
    if state1.num_modes != state2.num_modes:
        raise ValueError('Number of modes is not the same in both states.')

    N = state1.num_modes

    weights1 = state1.weights()
    weights2 = state2.weights()
    
    deltas = state1.means()[:,np.newaxis,:] - state2.means()[np.newaxis,:,:]

    covsum = state1.covs() + state2.covs() #covs are the same shape, so no broadcasting needed
    
    covsum_inv = np.linalg.inv(covsum)
    
    exp_arg = np.einsum('...j,...jk,...k', deltas, covsum_inv, deltas)
    
    if MP:
        weights_reorder = (weights1[np.newaxis,:]*weights2[:,np.newaxis]).reshape(len(weights1)*len(weights2))
        
        exp_arg = [mp.mpc(i) for i in exp_arg.reshape(len(weights1)*len(weights2))]
        exp_val = [mp.exp(f'{-0.5*i}') for i in exp_arg]
        weighted_exp = (weights_reorder * sf.hbar ** N
                   * exp_val / np.sqrt( np.linalg.det(covsum)) )
        fidelity = mp.fsum(weighted_exp)
    
    #exp_arg_vector = np.vectorize(math.exp)
    
    #weighted_exp = ( state1.weights()[:,np.newaxis] * state2.weights()[np.newaxis,:] * sf.hbar ** N
                    #* exp_arg_vector(-0.5*exp_arg.real) / np.sqrt( np.linalg.det(covsum)) )

    
    else:
        
        weighted_exp = ( state1.weights()[:,np.newaxis] * state2.weights()[np.newaxis,:] * sf.hbar ** N
                   * np.exp( -0.5 * exp_arg) / np.sqrt( np.linalg.det(covsum)) )
                   
        fidelity = np.sum(weighted_exp).real 
    #fidelity = math.fsum(weighted_exp.reshape(len(state1.weights())*len(state2.weights())))
    
    return fidelity


def fidelity_with_wigner(W1, W2, xvec, pvec):
    """
    Calculate fidelity via explicit Wigner functions.
    
    Args:
        W1 (ndarray): Wigner mesh
        W2 (ndarray):
        xvec (ndarray): points in x
        pvec (ndarray): points in p
        
    Returns:
        (float): fidelity
    """
    delta_x = xvec[-1] - xvec[-2]
    delta_p = pvec[-1] - pvec[-2]
    
    return 2 * np.pi * sf.hbar * np.sum(W1*W2*delta_x*delta_p)
    
    
def fidelity_wigner_single_mode(state1, state2, xvec, pvec):
    """
    Calculate fidelity of two (single mode) states using their Wigner functions. 
    At least one state must be pure.
    
    Args:
        state1 (BaseBosonicState):
        state2 (BaseBosonicState):
        xvec (ndarray): x points
        pvec (ndarray): p points
        
    Returns:
        (float): fidelity
    """
    
    #purity check
    if np.round(state1.purity(), 4) < 1 and np.round(state2.purity(), 4) < 1:
        raise ValueError('Both states are not pure.')
    #mode number check
    if state1.num_modes > 1 or state2.num_modes > 1:
        raise ValueError('One or more states have multiple modes.')
        
    wig1 = state1.wigner(mode = 0, xvec = xvec, pvec = pvec)
    wig2 = state2.wigner(mode = 0, xvec = xvec, pvec = pvec)
    
    delta_x = xvec[-1] - xvec[-2]
    delta_p = pvec[-1] - pvec[-2]

    return 2 * np.pi * sf.hbar * np.sum(wig1 * wig2 * delta_x * delta_p)

def fidelity_gkp(state, gkp):
    """
    Calculate fidelity of a state and a gkp state. Specialised for gkp to minimze np.linalg.inv time.
    
    Args:
        state (BaseBosonicState):
        gkp (BaseBosonicState):
        
    Returns:
        fidelity (float): fidelity of state with a gkp
    
    """
    
    #Equal number of modes check
    if state.num_modes != gkp.num_modes:
        raise ValueError('Number of modes is not the same in both states.')

    N = state.num_modes
    
    deltas = state.means()[:,np.newaxis,:] - gkp.means()[np.newaxis,:,:]

    covsum = state.covs() + gkp.covs()[0,:] #gkp covs are the same, so no need to use all of them  
    
    covsum_inv = np.linalg.inv(covsum)
    
    exp_arg = np.einsum('...j,...jk,...k', deltas, covsum_inv[:,np.newaxis,:,:], deltas)
    
    weighted_exp = ( state.weights()[:,np.newaxis] * gkp.weights()[np.newaxis,:] * sf.hbar ** N
                    * np.exp( -0.5 * exp_arg) / np.sqrt( np.linalg.det(covsum[:,np.newaxis,:,:])) )
    

    fidelity = np.sum(weighted_exp).real

    return fidelity


def fidelity_bosonic_pure(state1, state2, tol = 1e-15):
    """
    Calculate fidelity of two states when at least one of them is pure. 
    
    If state1 and state2 are the same state, and it is not pure, then this gives the purity. 
    If state1 and state2 are impure, but different, it gives the overlap.
    
    OBS: np.linalg.inv(covsum) slows things down.
    
    Args:
        state1 (BaseBosonicState):
        state2 (BaseBosonicState):
        
    Returns:
        fidelity (float): fidelity, purity or overlap depending on the situation.
    
    """
    
    #Equal number of modes check
    if state1.num_modes != state2.num_modes:
        raise ValueError('Number of modes is not the same in both states.')
    
    N = state1.num_modes
    
    deltas = state1.means()[:,np.newaxis,:] - state2.means()[np.newaxis,:,:]
    
    covsum = state1.covs()[:,np.newaxis,:,:] + state2.covs()[np.newaxis,:,:,:]
   
    exp_arg = np.einsum('...j,...jk,...k', deltas, np.linalg.inv(covsum), deltas)

    weighted_exp = ( state1.weights()[:,np.newaxis] * state2.weights()[np.newaxis,:] * sf.hbar ** N
                    * np.exp( -0.5 * exp_arg) / np.sqrt( np.linalg.det(covsum)) )
    

    fidelity = np.sum(weighted_exp)
    
    # Numerical error can yield fidelity marginally greater than 1
    if 1 - fidelity < 0 and fidelity - 1 < tol:
        fidelity = 1
        
    return fidelity.real


def Hilbert_Schmidt_distance(state1,state2):
    """
    Calculate the Hilbert Schmidt distance, Tr( (state1 - state2) ^ 2 ).
    Args:
        state1 (BaseBosonicState):
        state2 (BaseBosonicState):
        
    Returns:
        (float): HS distance
    """
    return state1.purity() + state2.purity() - 2*fidelity_bosonic_pure(state1,state2)


def gkp_fidelity_to_sq_vacuum(gkp, epsilon):
    """ Get fidelity of gkp state to squeezed vacuum with squeezing equal to epsilon value.
    
    Args:
        gkp (BaseBosonicState or None): 
        epsilon (float): damping of gkp
    Returns:
        fid (float): fidelity
    """
    
    if gkp == None:
        weights_gkp, means_gkp, covs_gkp = prepare_gkp(np.array([0,0]), epsilon = epsilon)
        gkp = BaseBosonicState([means_gkp.real, covs_gkp.real, 
                                weights_gkp.real],num_modes = 1, num_weights = len(weights_gkp))
    
    r = dB_to_r(epsilon_to_dB(epsilon))
    

    fid = fidelity_gkp(squeezed_state(r), gkp).real
    
    #fid = fidelity_bosonic_pure(gkp, squeezed_state(r)).real
    return fid