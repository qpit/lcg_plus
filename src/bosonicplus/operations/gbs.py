import numpy as np
#from thewalrus.symplectic import xpxp_to_xxpp, xxpp_to_xpxp, expand, beam_splitter, rotation, squeezing, expand_vector
from bosonicplus.base import State
from bosonicplus.operations.symplectic import *
from copy import copy

def build_interferometer(params : dict, nmodes : int, out = False, setting = 'no_phase'):
    """
    Build nmode interferometer with settings given by params dict
    
    Args: 
        params (dict):
        nmodes (int): number of modes
        
    Returns:
        state (bosonicplus.State):
    """

    state = State(nmodes) #ordering is xxpp

    
    Stot = np.eye(2*nmodes) 
    sqz = params['sqz']

    #Depending on the circuit size, it might be faster to simply apply the gates directly onto the state
    #What we do here is to build up the total symplectic, then apply it to covs, means (in xxpp) 
    
    #Squeezing symplectics
    for i in range(len(sqz)):
        S = expand_symplectic_matrix(squeezing(sqz[i][0], sqz[i][1]), [i], nmodes)
        #S = expand(squeezing(sqz[i][0], sqz[i][1]), i, nmodes)
        #S = xxpp_to_xpxp(expand(squeezing(sqz[i][0], sqz[i][1]), sqz[i][2], nmodes))
        Stot = S @ Stot

        if out:
            print('Sgate[{:.3f},{:.3f}] on mode {}'.format(sqz[i][0],sqz[i][1], i))

    #Beamsplitter/Two-mode-squeezing symplectics
    bs = params['bs']
    for i in range(len(bs)):
        if setting == 'two_mode_squeezing':
        
            BS = expand_symplectic_matrix(two_mode_squeezing(bs[i][0], bs[i][1]),bs[i][2], nmodes)
            if out:
                print('TMSgate[{:.3f},{:.3f}] on modes {} and {}'.format(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1]))
        else:
            BS = expand_symplectic_matrix(beam_splitter(bs[i][0], bs[i][1]),bs[i][2], nmodes)
            if out:
                print('BSgate[{:.3f},{:.3f}] on modes {} and {}'.format(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1]))
        #BS = xxpp_to_xpxp(expand(beam_splitter(bs[i][0], bs[i][1]),bs[i][2], nmodes))
        #BS = expand(beam_splitter(bs[i][0], bs[i][1]),bs[i][2], nmodes)
        Stot  = BS @ Stot
        #circuit.beamsplitter(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1])
        
        
    #Additional rotation symplectics
    phis = params['phis']
    if len(phis) != 0: 
        for i in range(len(phis)):
            #S = xxpp_to_xpxp(expand(rotation(phis[i][0]), phis[i][1], nmodes))
            S = expand_symplectic_matrix(rotation(phis[i][0]), [i], nmodes)
            #S = expand(rotation(phis[i][0]), i, nmodes)
            Stot = S @ Stot
            if out:
                print('Rgate[{:.3f}] on mode {}'.format(phis[i][0], i) )

    #Apply the symplectic to the state
    #state.apply_symplectic(xxpp_to_xpxp(Stot))
    state.apply_symplectic(Stot)
                

    #Apply displacements at the end of the circuit if any
    alphas = params['alphas']
    

    if len(alphas) != 0:
        disp = np.zeros(2*nmodes)
        for i in range(len(alphas)):
            beta = alphas[i][0]
            mode = alphas[i][1]
            #disp += xxpp_to_xpxp(expand_vector(alpha, alpha[i][1])) #hbar = 2 by default
            disp += expand_displacement_vector(beta, mode, nmodes) #hbar = 2 by default
            if out:
                print(r'Dgate[{:.3f}] on mode {}'.format(beta, mode))
        
        #state.apply_displacement(xxpp_to_xpxp(disp))
        state.apply_displacement(disp)
    
        
    return state


def bosonicplus_circuit(nmodes, r, eta, n, fast = True):
    state = State(nmodes)
    bs = xxpp_to_xpxp(beam_splitter(np.pi/4,0))

    for i in range(nmodes):
        
        state.apply_symplectic_fast(xxpp_to_xpxp(squeezing(r, i*np.pi)),[i])
    for i in range(nmodes):
        if i < nmodes-1:
            state.apply_symplectic_fast(bs, [i,i+1])
    if eta != 1:
        state.apply_loss(np.repeat(eta,nmodes),np.zeros(nmodes))
    
    for i in range(nmodes-1):
        state.post_select_fock_coherent(0,n[i],inf=1e-4,red_gauss = fast)
        print(f'detecting {n[i]} photons')
        print('no. of weights', state.num_weights)
        #state.post_select_fock_coherent(0,n)
    
    return state

def build_interferometer_gradients(params : dict, nmodes : int, out = False, setting = 'no_phase'):
    """
    Build nmode interferometer with settings given by params dict. Generate gradients of covariance matrix, and
    displacement vector.
    
    Args: 
        params (dict):
        nmodes (int): number of modes
        
    Returns:
        state (bosonicplus.State):
    """

    state = State(nmodes)
    
    Stot = np.eye(2*nmodes) 
    sqz = params['sqz']

    Slist = []
    Glist = []

    Sgrad = np.eye(2*nmodes)
    
    #Squeezing symplectics
    for i in range(len(sqz)):
        S = expand_symplectic_matrix(squeezing(sqz[i][0], sqz[i][1]), [i], nmodes)

        Slist.append(S)
        G1, G2 = squeezing_gradients(sqz[i][0], sqz[i][1])
        
        G1 = expand_symplectic_gradient(G1, [i], nmodes)
        #Ignore G2 for now
        Glist.append(G1)
        
        if out:
            print('Sgate[{:.3f},{:.3f}] on mode {}'.format(sqz[i][0],sqz[i][1], i))

    #Beamsplitter symplectics
    bs = params['bs']
    for i in range(len(bs)):
        if setting == 'two_mode_squeezing':
            BS = expand_symplectic_matrix(two_mode_squeezing(bs[i][0], bs[i][1]),bs[i][2], nmodes)
            G1, G2 = two_mode_squeezing_gradients(bs[i][0], bs[i][1])

            G1 = expand_symplectic_gradient(G1, bs[i][2], nmodes)
            G2 = expand_symplectic_gradient(G2, bs[i][2], nmodes)
    
            Glist.append(G1)
            Glist.append(G2)
            if out:
                print('TMSgate[{:.3f},{:.3f}] on modes {} and {}'.format(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1]))
        elif setting =='no_phase':
            
        #BS = xxpp_to_xpxp(expand(beam_splitter(bs[i][0], bs[i][1]),bs[i][2], nmodes))
            BS = expand_symplectic_matrix(beam_splitter(bs[i][0], bs[i][1]), bs[i][2], nmodes)
    
            G1, G2 = beam_splitter_gradients(bs[i][0], bs[i][1])
                                         
            G1 = expand_symplectic_gradient(G1, bs[i][2], nmodes)
    
            Glist.append(G1)
            if out:
                print('BSgate[{:.3f},{:.3f}] on modes {} and {}'.format(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1]))
        
        Slist.append(BS)
        
        
        
    #Additional rotation symplectics if any
    phis = params['phis']
    if len(phis) !=0: 
        for i in range(len(phis)):
            #S = xxpp_to_xpxp(expand(rotation(phis[i][0]), phis[i][1], nmodes))
            S = expand_symplectic_matrix(rotation(phis[i][0]), [i], nmodes)
            Slist.append(S)

            G1 = expand_symplectic_gradient(rotation_gradient(phis[i][0]),[i],nmodes)
            Glist.append(G1)
            
            if out:
                print('Rgate[{:.3f}] on mode {}'.format(phis[i][0], i) )

    #Apply the symplectic to the state

    Stot = multiply_matrices(Slist[::-1]) #Reverse the order and multiply matrices together
    
    state.apply_symplectic(Stot)
                
    #Apply displacements at the end of the circuit if any
    alphas = params['alphas']

    if len(alphas)!=0:
        raise ValueError('Gradients of displacements not implemented.')
        disp = np.zeros(2*nmodes)
        for i in range(len(alphas)):
            beta = alphas[i][0]
            mode = alphas[i][1]
            #disp += xxpp_to_xpxp(expand_vector(alpha, alpha[i][1])) #hbar = 2 by default
            disp += expand_displacement_vector(beta, mode, nmodes) #hbar = 2 by default
            if out:
                print(r'Dgate[{:.3f}] on mode {}'.format(beta, mode))
        
        #state.apply_displacement(xxpp_to_xpxp(disp))
        state.apply_displacement(disp)
    

    #Obtain the partial derivative of the symplectic matrix

    dS = []
    for i, g in enumerate(Glist):
        S = copy(Slist) 
        if setting == 'two_mode_squeezing':
        
            S[int(i/2)] = g #Replace i'th symplectic matrix with it's partial derivative
        else:
            S[i] = g #Replace i'th symplectic matrix with it's partial derivative
        
        dS.append(multiply_matrices(S[::-1]))

    
    cov_gradient = 0.5 * state.hbar * np.array([Stot @ i.T + i @ Stot.T for i in dS])[:,np.newaxis,:,:] #Broadcast weight dimension
    mean_gradient = np.zeros((len(Glist), 1, nmodes*2)) 
    weights_gradient = np.zeros((len(Glist),1))
        
    state.update_gradients([mean_gradient, cov_gradient, weights_gradient])
    return state