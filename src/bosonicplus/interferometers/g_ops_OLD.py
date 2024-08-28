import strawberryfields as sf
import numpy as np
from strawberryfields.backends.bosonicbackend import ops
from strawberryfields.backends.bosonicbackend.bosoniccircuit import from_xp, to_xp
from strawberryfields.backends.bosonicbackend.bosoniccircuit import BosonicModes
from strawberryfields.backends.states import BaseBosonicState
import thewalrus.symplectic as symp
from strawberryfields.decompositions import williamson, bloch_messiah, rectangular_phase_end

from math import factorial, fsum

from bosonicplus.conversions import *

# GAUSSIAN OPERATIONS
# ------------------------------------
def beamsplitter(theta, phi, k, l, n):
    r"""Beam splitter symplectic between modes k and l for n mode system.
    Args:
        theta (float): real beamsplitter angle
        phi (float): complex beamsplitter angle
        k (int): first mode
        l (int): second mode
        n (int): total number of modes
    Raises:
        ValueError: if the first mode equals the second mode
    Returns:
        bs (array): n mode symplectic matrix in xpxp notation
    """

    if k == l:
        raise ValueError("Cannot use the same mode for beamsplitter inputs.")

    bs = symp.expand(symp.beam_splitter(theta, phi), [k, l], n)
    bs = symp.xxpp_to_xpxp(bs) #convert to xpxp notation

    return bs
    
def squeeze(r, phi, k, n):
    r"""Symplectic for squeezing mode ``k`` by the amount ``r``.
    Args:
        r (float): squeezing magnitude
        k (int): mode to be squeezed
        n (int): total number of modes
    Returns:
        sq (array): n mode squeezing symplectic matrix in xpxp notation
    """

    sq = symp.expand(symp.squeezing(r,phi), k, n)
    sq = symp.xxpp_to_xpxp(sq)

    return sq
    
def rotation(theta, k, n):
    r"""Symplectic for a rotation operation on mode k.
        Args:
            theta (float): rotation angle
            k (int): mode to be rotated
            n (int): total number of modes
        Returns:
            sr (array): n mode rotation symplectic matrix in xpxp notation
    """
    sr = symp.expand(symp.rotation(theta), k, n)
    sr = symp.xxpp_to_xpxp(sr)
    return sr

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
    


# BUILDING AN INTERFEROMETER
# ----------------------------------------

def gen_Clements_indices(nmodes):
    """ Get a list of the beamsplitter indices of a Clements interferometer.
    Adaptation of https://github.com/XanaduAI/approximate-GKP-prep/blob/master/StatePrepCircuits.ipynb
    
    """
    ind = []
    k = 0
    for i in range(nmodes):
        if i % 2 == 0:
            for j in range(int(np.floor(nmodes/2))):
                ind.append([2*j, 2*j+1])
                k += 1
        else:
            for j in range(1, int(np.floor((nmodes-1)/2)) + 1):
                ind.append([2*j-1, 2*j])
                k += 1
    return ind

def gen_cascade_indices(nmodes):
    """ Get a list of the beamsplitter indices of a cascaded interferometer
    """
    ind = []
    
    for i in range(nmodes-1):
        ind.append([i,i+1])
        
    return ind

def gen_inv_cascade_indices(nmodes):
    """ Get a list of the beamsplitter indices of an inverted cascaded interferometer
    """

    ind = gen_cascade_indices(nmodes)
    ind.reverse()

    return ind
        

def gen_interferometer_params(nmodes, r_max_dB, bs_arrange = 'Clements'):
    """
    Generate random interferometer parameters in Clements decomp.
    
    Args: 
        nmodes (int): number of modes
        r_max_dB (float): max squeezing in dB
    Returns:
        params (dict): dictionary
    """

    # Squeezers
    r = dB_to_r(r_max_dB)
    rs = np.random.uniform(0.1,r,nmodes)
    rs_angle = np.random.uniform(-np.pi,np.pi, nmodes)
    sqz = list(zip(rs, rs_angle))

    # Beamsplitters
    if bs_arrange == 'Clements':
        inds = gen_Clements_indices(nmodes)
    elif bs_arrange == 'cascade':
        inds = gen_cascade_indices(nmodes)
    elif bs_arrange == 'inv_cascade':
        inds = gen_inv_cascade_indices(nmodes)
    else:
        raise ValueError('bs_arrange must be either str(Clements) or str(cascade).')
        
    nbs = len(inds)
    thetas = np.random.uniform(0.1,np.pi/2-0.1,nbs)
    phis = np.random.uniform(-np.pi,np.pi,nbs)
    bs = list(zip(thetas, phis, inds))

    # Extra phases
    #phis_extra = np.random.uniform(-np.pi,np.pi,nmodes) 
    phis_extra = None

    # Loss right before pnrd
    loss = None
    #loss = np.repeat(0.5,nmodes)
    alpha = None

    params = {'sqz': sqz, 'bs': bs, 'phis':  phis_extra, 'loss': loss, 'alpha': alpha}
    
    return params

def gen_Clements_params(nmodes, r_max_dB):
    """
    Generate random interferometer parameters in Clements decomp.
    
    Args: 
        nmodes (int): number of modes
        r_max_dB (float): max squeezing in dB
    Returns:
        params (dict): dictionary
    """

    # Squeezers
    r = dB_to_r(r_max_dB)
    rs = np.random.uniform(0.1,r,nmodes)
    rs_angle = np.random.uniform(-np.pi,np.pi, nmodes)
    sqz = list(zip(rs, rs_angle))

    # Beamsplitters
    inds = gen_Clements_indices(nmodes)
    nbs = len(inds)
    thetas = np.random.uniform(0.1,np.pi/2-0.1,nbs)
    phis = np.random.uniform(-np.pi,np.pi,nbs)
    bs = list(zip(thetas, phis, inds))

    # Extra phases
    #phis_extra = np.random.uniform(-np.pi,np.pi,nmodes) 
    phis_extra = None

    # Loss right before pnrd
    loss = None
    #loss = np.repeat(0.5,nmodes)
    alpha = None

    params = {'sqz': sqz, 'bs': bs, 'phis':  phis_extra, 'loss': loss, 'alpha': alpha}
    
    return params


def build_interferometer(params, nmodes, out = False):
    """
    Build nmode interferometer with settings given by params dict
    
    Args: 
        params (dict):
        nmodes (int): number of modes
        
    Returns:
        circuit (BosonicModes):
    """
    circuit = BosonicModes(num_subsystems = nmodes)
    
    #Apply squeezing symplectics 
    for i in range(nmodes):
        circuit.squeeze(params['sqz'][i][0], params['sqz'][i][1],i)
        
        if out:
            print('Sgate[{:.3f},{:.3f}] on mode {}'.format(params['sqz'][i][0],params['sqz'][i][1], i))

    #Apply beamsplitters 
    bs = params['bs']
    for i in range(len(bs)):
        circuit.beamsplitter(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1])
        if out:
            print('BSgate[{:.3f},{:.3f}] on modes {} and {}'.format(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1]))
        
    #Apply additional rotation
    if params['phis']: 
        for i in range(nmodes):
            circuit.phase_shift(params['phis'][i], i)
            if out:
                print('Rgate[{:.3f}] on mode {}'.format(params['phis'][i],i) )
                
    #Apply loss if any
    if params['loss']:
        for i in range(nmodes):
            circuit.loss(1-params['loss'][i],i)
            if out:
                print(r'{:.1f}% loss channel on mode {}'.format(params['loss'][i]*100,i)) 

    #Apply displacements at the end of the circuit if any
    if params['alpha']:
        for i in range(nmodes):
            alpha = params['alpha'][i]
            circuit.displace(np.abs(alpha), np.angle(alpha), i)
            if out:
                print(r'Dgate[{:.3f}] on mode {}'.format(alpha, i))
                
    circuit.success = 1  #Attribute for measurement probabilities
        
    return circuit

def unitary_xp2a(U):
    """Convert 2Nx2N unitary matrix in the xxxppp representation to a 
    NxN unitary matrix operating on annihilation operators.
    """
    n = U.shape[0] // 2
    In = np.identity(n)
    T = .5 * np.block([[In, 1j*In], [In, -1j*In]])
    Ua = T @ U @ np.linalg.inv(T)
    return Ua[:n, :n]


# Balanced beamsplitter network
# ----------------------------------------

def symmetric_multiport_interferometer_unitary(N):
    """Unitary for a symmetric multiport interferometer
    from Eq. 18-19 of https://journals.aps.org/pra/pdf/10.1103/PhysRevA.55.2564

    Equal probability of each input mode going to N output ports
    
    Args:
        N (int): number of modes
    Returns:
        U ((N,N) complex array): unitary
    """
    U = np.zeros((N,N), dtype = 'complex')
    gamma_N = np.exp(1j * 2 * np.pi/N)

    for i in np.arange(N):
        for j in np.arange(N):
            U[i,j] = 1/np.sqrt(N) * gamma_N ** (i*j)
    return U

def symmetric_multiport_interferometer_symplectic(N):
    """Symplectic for a symmetric multiport interferometer

        S = [[X, -Y],[Y, X]] where X = U.real and Y = U.imag where U is the unitary matrix
    
    Args:
        N (int): number of modes
    Returns:
        S ((2N,2N) array): symplectic matrix in xpxp
    """
    S = symp.interferometer(symmetric_multiport_interferometer_unitary(N)) #in xxpp
    return symp.xxpp_to_xpxp(S) #to xpxp


def build_symmetric_interferometer(params, nmodes, out = False):
    """
    Build nmode symmetric interferometer with equal initial squeezing.
    Args: 
        params (dict): 
        nmodes (int): number of modes
        loss (ndarray or None): loss at the end of the circuit 
        out (bool): 
        
    Returns:
        circuit (BosonicModes):
        tlist (ndarray): beamsplitter index and parameter list for visualisation
        phis_extra (ndarray): additional rotations if any for visualisation
    """
    
    circuit = BosonicModes(num_subsystems = nmodes)
    
    #Apply squeezing symplectics 
    for i in range(nmodes):
        circuit.squeeze(params['sqz'][i][0],params['sqz'][i][1],i)
        
        if out:
            print('Sgate[{:.3f},{:.3f}] on mode {}'.format(params['sqz'][i][0],params['sqz'][i][1],i))

    covs = circuit.covs
    
    #Apply symmetric interferometer
    U = symmetric_multiport_interferometer_unitary(nmodes)

    circuit.apply_u(U) 
    if out: 
        print('Applied symmetric multiport interferometer unitary.')
                
    #Apply loss if any
    if params['loss'] != None:
        for i in range(len(params['loss'])):
            circuit.loss(1-params['loss'][i],i)
            if out:
                print('{:.1f}% loss channel on mode {}'.format(params['loss']*100,i)) 
                
    circuit.success = 1
    return circuit

# OLD
# ------------------------------

def get_covs(params, nmodes):
    """Returns the covariance matrix of the GBS circuit.

    Args:
        params (dict): 
        nmodes (int): number of modes
    Returns:
        covs (ndarray): covariance matrix 
    """
    circuit = build_interferometer(params, nmodes)
    return circuit.covs
    
def demultiplex(circuit, mode, M, out = False):
    """
    Demultiplex mode into M branches by interfering mode with M-1 ancillary vacuum modes. 
    
    Args:
        circuit (object): BosonicModes class
        mode (int): mode index to be demultiplexed
        M (int): number of branches
        out (bool):
        
    Returns: 
        circuit (object): updated circuit with ancillary modes
        modes_det (list): indices if demultiplexing modes
        
    """
    if out:
        print('Demultiplexing mode {} into {} branches.'.format(mode, M))
        
    nmodes = circuit.nlen
    
    #Add ancilla modes
    for i in range(M-1):
        circuit.add_mode()
    
    #Implement demultiplexing beamsplitters
    for i in range(M-1):
        idx = i + nmodes
        theta = np.arccos(np.sqrt( (M - (i+1))/(M - i)))
        circuit.beamsplitter(theta,0, mode, idx)
        if out == True:
            print('BSgate[{}/{}] on modes {} and {}'.format(M-(i+1), M-i, mode, idx))
            
    #Detection modes
    modes_det = [mode] + [x for x in range(nmodes, nmodes + M-1)]
    
    if out:
        
        photons = average_photon(circuit)
        print('Average photon number in split modes: \n {}'.format(photons[modes_det]))
        vac_f = np.zeros(len(modes_det))
        for i, m in enumerate(modes_det):
            vac_f[i] = np.abs(circuit.fidelity_vacuum([m]))

        print('Vacuum fidelity, {}'.format(vac_f))
        print('Detection on {} modes.'.format(modes_det))
        
    return circuit, modes_det

