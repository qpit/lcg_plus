import numpy as np
from strawberryfields.backends.bosonicbackend.bosoniccircuit import BosonicModes
from .symplectics import symmetric_multiport_interferometer_unitary


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
    sqz = params['sqz']
    for i in range(len(sqz)):
        circuit.squeeze(sqz[i][0], sqz[i][1], sqz[i][2])
        
        if out:
            print('Sgate[{:.3f},{:.3f}] on mode {}'.format(sqz[i][0],sqz[i][1], sqz[i][2]))

    #Apply beamsplitters 
    bs = params['bs']
    for i in range(len(bs)):
        circuit.beamsplitter(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1])
        if out:
            print('BSgate[{:.3f},{:.3f}] on modes {} and {}'.format(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1]))
        
    #Apply additional rotation
    phis = params['phis']
    if phis: 
        for i in range(len(phis)):
            circuit.phase_shift(phis[i][0], phis[i][1])
            if out:
                print('Rgate[{:.3f}] on mode {}'.format(phis[i][0], phis[i][1]) )
                
    #Apply loss if any
    loss = params['loss']
    if loss:
        for i in range(len(loss)):
            circuit.loss(1-loss[i][0],loss[i][1])
            if out:
                print(r'{:.1f}% loss channel on mode {}'.format(loss[i][0]*100,loss[i][1])) 

    #Apply displacements at the end of the circuit if any
    alphas = params['alpha']
    if alphas:
        for i in range(len(alphas)):
            alpha = alpha[i][0]
            circuit.displace(np.abs(alpha), np.angle(alpha), alphas[i][1])
            if out:
                print(r'Dgate[{:.3f}] on mode {}'.format(alpha, alphas[i][1]))
                
    circuit.success = 1  #Attribute for measurement probabilities
        
    return circuit

# Balanced beamsplitter network
# ----------------------------------------

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
    
    sqz = params['sqz']
    for i in range(len(sqz)):
        circuit.squeeze(sqz[i][0], sqz[i][1], sqz[i][2])
        
        if out:
            print('Sgate[{:.3f},{:.3f}] on mode {}'.format(sqz[i][0],sqz[i][1], sqz[i][2]))

    covs = circuit.covs
    
    #Apply symmetric interferometer
    U = symmetric_multiport_interferometer_unitary(nmodes)

    circuit.apply_u(U) 
    if out: 
        print('Applied symmetric multiport interferometer unitary.')
                
   #Apply loss if any
    loss = params['loss']
    if loss:
        for i in range(len(loss)):
            circuit.loss(1-loss[i][0],loss[i][1])
            if out:
                print(r'{:.1f}% loss channel on mode {}'.format(loss[i][0]*100,loss[i][1])) 
                
    circuit.success = 1
    return circuit

# OLD UNUSED FUNCTIONS
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

