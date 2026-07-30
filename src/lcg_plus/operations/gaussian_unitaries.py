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
from scipy.sparse import issparse
from functools import reduce
from thewalrus.symplectic import xxpp_to_xpxp
from lcg_plus.helper.from_sf import chop_in_blocks_multi, chop_in_blocks_vector_multi

hbar = 2

def symplectic_form(num_modes):
    Omega = np.array([[0,1],[-1,0]]) #Single mode 
    return np.kron(np.eye(num_modes),Omega)

def is_symplectic(mat, rtol=1e-05, atol=1e-08):
    """Check if mat is symplectic"""
    m, n = mat.shape

    if n != m:
        return False
    if n % 2 != 0:
        return False
    
    num_modes = n // 2
    Omega = symplectic_form(num_modes)
    return np.allclose(Omega, mat @ Omega @ mat, rtol=rtol, atol=atol)

def apply_symplectic_on_subsystem(means, covs, num_modes, num_weights, symp_mat, modes_subsystem):
    """Apply a symplectic transform (in xpxp ordering) on subsystem elements of the covariance matrices and displacement vectors,
    then reassemble the covariance matrices and disp vectors from the updated elements. 
    This method has better performance than apply_symplectic() when the number of modes is large (> 20). 
    """

    if len(modes_subsystem) != symp_mat.shape[0]/2:
        raise ValueError('Symplectic matrix must have same dimension as the modes list')
    if num_modes == 2 and len(modes_subsystem) == 2:
        raise ValueError('Use apply_symplectic.')
    
    mode_ind_subsystem = np.concatenate((2 * np.array(modes_subsystem), 2 * np.array(modes_subsystem) + 1)) #in xxpp 
    mode_ind_subsystem = xxpp_to_xpxp(mode_ind_subsystem) #back to xpxp
    
    mode_ind_total = np.arange(2*num_modes)
    
    mode_ind_rest = list(set(mode_ind_total) - set(mode_ind_subsystem))
    
    A, AB, B = chop_in_blocks_multi(covs, mode_ind_subsystem) #Chop the system into A modes: untouched, and B modes: the modes the symplectic is applied to
    a, b = chop_in_blocks_vector_multi(means, mode_ind_subsystem)
    
    Bnew = np.einsum("...jk,...kl,...lm",symp_mat,B,symp_mat.T)
    ABnew = np.einsum("...jk,...kl",AB,symp_mat.T)
    bnew = np.einsum("...jk,...k",symp_mat,b)
    
    nw = num_weights #Number of weights
    if covs.shape[0] != 1:
        covs[np.ix_(np.arange(nw),mode_ind_subsystem,mode_ind_subsystem)] = Bnew 
        covs[np.ix_(np.arange(nw),mode_ind_rest, mode_ind_subsystem)] = ABnew
        covs[np.ix_(np.arange(nw),mode_ind_subsystem, mode_ind_rest)] = np.transpose(ABnew, axes = [0,2,1])
    else:
        covs[np.ix_([0],mode_ind_subsystem,mode_ind_subsystem)] = Bnew 
        covs[np.ix_([0],mode_ind_rest, mode_ind_subsystem)] = ABnew
        covs[np.ix_([0],mode_ind_subsystem, mode_ind_rest)] = np.transpose(ABnew, axes = [0,2,1])
    
    means[np.ix_(np.arange(nw),mode_ind_subsystem)] = bnew
    return means, covs

def apply_symplectic_full(means, covs, symp_mat : np.ndarray):
    """ Apply symplectic transform (in xpxp ordering) on the covariance matrices and displacement vectors,
        V -> S V S.T
        r -> S r
    Note that symplectics from thewalrus are in xxpp ordering and we use xpxp ordering.
        symp_mat (array) : symplectic matrix
    """
    
    #First check that symp_mat has the correct dimensions
    if np.shape(symp_mat)[0] != int(np.shape(means)[-1]):
        raise ValueError('symp_mat must must be 2nmodes x 2nmodes.')
    
    if covs.shape[0] == 1: 
        new_covs = symp_mat @ covs @ symp_mat.T
    else:
        #Didn't test if this works
        new_covs = np.einsum("...jk,...kl,...lm", symp_mat, covs, symp_mat.T)

    new_means = np.einsum("...jk,...k", symp_mat, means)

    return new_means, new_covs


def beam_splitter(theta, phi):
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    cs = np.cos(theta) * np.eye(2)
    ss = np.sin(theta) * np.array([[cp, -sp], [sp, cp]])
    return np.block([[cs, -ss.T],[ss, cs]])

def rotation(phi):
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([[c, -s],[s,c]])

def rotation_gradient(phi):
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([[-s, -c],[c,-s]])

def beam_splitter_gradients(theta, phi):
    """Get partial deriviatives of the symplectic matrix wtr theta and phi
    Returns
        dS/dtheta, dS/dphi
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    cs = -np.sin(theta) * np.eye(2)
    ss = np.cos(theta) * np.array([[cp, -sp], [sp, cp]])

    null = np.zeros((2,2))
    sphi = np.sin(theta) * np.array([[-sp, -cp],[cp, -sp]])
    
    return np.block([[cs, -ss.T],[ss, cs]]), np.block([[null, -sphi.T], [sphi, null]])

def squeezing(r, phi):
    s = np.sin(phi)
    c = np.cos(phi)
    
    return np.cosh(r)*np.eye(2) - np.sinh(r)*np.array([[c, s],[s,-c]])
    
def squeezing_gradients(r, phi):
    """
    Returns
        dS/dr, dS/dphi
    """
    s = np.sin(phi)
    c = np.cos(phi)
    return np.sinh(r)*np.eye(2)-np.cosh(r)*np.array([[c,s],[s,-c]]), -np.sinh(r)*np.array([[-s,c],[c,s]])

def two_mode_squeezing(r, phi):
    c = np.cos(phi)
    s = np.sin(phi)

    Sphi = np.array([[c, s], [s, -c]])
    return np.block([[np.cosh(r)*np.eye(2), np.sinh(r) * Sphi], [np.sinh(r)*Sphi, np.cosh(r)*np.eye(2)]])

def two_mode_squeezing_gradients(r,phi):
    """
    Returns
        dS/dr, DS/dphi
    """
    c = np.cos(phi)
    s = np.sin(phi)
    Sphi = np.array([[c, s], [s, -c]])
    dSphi = np.array([[-s, c],[c, s]])
    ss = np.sinh(r)*np.eye(2)
    cs = -np.cosh(r)*Sphi
    null = np.zeros((2,2))
    return np.block([[ss, cs],[cs,ss]]), np.block([[null, np.sinh(r)*dSphi],[np.sinh(r)*dSphi, null]])
    
    
def disp_gradients(r, phi):
    """
    Returns
        dS/dr, dS/dphi
    """
    c = np.cos(phi)
    s = np.sin(phi)
    return np.sqrt(hbar*2) * np.array([c,s]), np.sqrt(hbar*2)*r*np.array([-s,c])
    
def expand_symplectic_matrix(S, target_modes, total_modes):
    """
    Written by ChatGPT.
    Embed a smaller 2N x 2N symplectic matrix into a larger 2M x 2M symplectic matrix.

    Parameters:
    - S (np.ndarray): 2N x 2N symplectic matrix acting on `len(target_modes)` modes.
    - target_modes (list of int): The modes (0-indexed) that S acts on.
    - total_modes (int): Total number of modes M for the output (resulting in 2M x 2M matrix).

    Returns:
    - np.ndarray: The expanded 2M x 2M symplectic matrix.
    """
    N = len(target_modes)
    assert S.shape == (2*N, 2*N), "Input matrix must be 2N x 2N"
    
    M = total_modes
    S_expanded = np.eye(2 * M)

    for i, mode_i in enumerate(target_modes):
        for j, mode_j in enumerate(target_modes):
            # Extract 2x2 block from S
            block = S[2*i:2*i+2, 2*j:2*j+2]
            # Place it in the expanded matrix
            S_expanded[2*mode_i:2*mode_i+2, 2*mode_j:2*mode_j+2] = block

    return S_expanded

def expand_displacement_vector(d, target_modes, total_modes):
    """
    Written by ChatGPT.
    Embed a 2N-dimensional displacement vector into a 2M-dimensional vector.

    Parameters:
    - d (np.ndarray): Displacement vector of length 2N.
    - target_modes (list of int): Modes that d acts on.
    - total_modes (int): Total number of modes M.

    Returns:
    - np.ndarray: Expanded 2M-dimensional displacement vector.
    """
    N = len(target_modes)
    assert d.shape == (2 * N,), "Displacement vector must have length 2N"
    
    D_expanded = np.zeros(2 * total_modes)

    for i, mode in enumerate(target_modes):
        D_expanded[2 * mode: 2 * mode + 2] = d[2 * i: 2 * i + 2]

    return D_expanded

def expand_symplectic_gradient(S, target_modes, total_modes):
    """
    Written by ChatGPT.
    Embed a smaller 2N x 2N symplectic gradient matrix into a larger 2M x 2M symplectic gradient matrix (with zero entries)

    Parameters:
    - S (np.ndarray): 2N x 2N symplectic matrix acting on `len(target_modes)` modes.
    - target_modes (list of int): The modes (0-indexed) that S acts on.
    - total_modes (int): Total number of modes M for the output (resulting in 2M x 2M matrix).

    Returns:
    - np.ndarray: The expanded 2M x 2M symplectic matrix.
    """
    N = len(target_modes)
    assert S.shape == (2*N, 2*N), "Input matrix must be 2N x 2N"
    
    M = total_modes
    S_expanded = np.zeros((2 * M,2 * M))

    for i, mode_i in enumerate(target_modes):
        for j, mode_j in enumerate(target_modes):
            # Extract 2x2 block from S
            block = S[2*i:2*i+2, 2*j:2*j+2]
            # Place it in the expanded matrix
            S_expanded[2*mode_i:2*mode_i+2, 2*mode_j:2*mode_j+2] = block

    return S_expanded


def multiply_matrices(matrix_list):
    """
    Written by ChatGPT
    Multiply a list of matrices in order: result = A @ B @ C @ ...
    
    Parameters:
    - matrix_list (list): List of matrices (np.ndarray or scipy.sparse matrices)
    
    Returns:
    - np.ndarray or scipy.sparse: The resulting matrix product.
    """
    if not matrix_list:
        raise ValueError("matrix_list must not be empty.")

    # Check if any matrix is sparse
    use_sparse = any(issparse(mat) for mat in matrix_list)

    # Reduce the list by successive multiplication
    result = reduce(lambda A, B: A @ B, matrix_list)

    return result
    
