import numpy as np
from thewalrus.symplectic import xpxp_to_xxpp, xxpp_to_xpxp, expand, beam_splitter, rotation, squeezing, expand_vector
from bosonicplus.newpackage.base import State

def build_interferometer(params : dict, nmodes : int, out = False):
    """
    Build nmode interferometer with settings given by params dict
    
    Args: 
        params (dict):
        nmodes (int): number of modes
        
    Returns:
        circuit (BosonicModes):
    """

    state = State(nmodes) #ordering is xxpp

    
    Stot = np.eye(2*nmodes) 
    sqz = params['sqz']

    #Depending on the circuit size, it might be faster to simply apply the gates directly onto the state
    #What we do here is to build up the total symplectic, then apply it to covs, means (in xxpp) 
    
    #Squeezing symplectics
    for i in range(len(sqz)):
        S = expand(squeezing(sqz[i][0], sqz[i][1]), sqz[i][2], nmodes)
        #S = xxpp_to_xpxp(expand(squeezing(sqz[i][0], sqz[i][1]), sqz[i][2], nmodes))
        Stot = S @ Stot

        if out:
            print('Sgate[{:.3f},{:.3f}] on mode {}'.format(sqz[i][0],sqz[i][1], sqz[i][2]))

    #Beamsplitter symplectics
    bs = params['bs']
    for i in range(len(bs)):
        #BS = xxpp_to_xpxp(expand(beam_splitter(bs[i][0], bs[i][1]),bs[i][2], nmodes))
        BS = expand(beam_splitter(bs[i][0], bs[i][1]),bs[i][2], nmodes)
        Stot  = BS @ Stot
        #circuit.beamsplitter(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1])
        if out:
            print('BSgate[{:.3f},{:.3f}] on modes {} and {}'.format(bs[i][0], bs[i][1], bs[i][2][0], bs[i][2][1]))
        
    #Additional rotation symplectics
    phis = params['phis']
    if phis: 
        for i in range(len(phis)):
            #S = xxpp_to_xpxp(expand(rotation(phis[i][0]), phis[i][1], nmodes))
            S = expand(rotation(phis[i][0]), phis[i][1], nmodes)
            Stot = S @ Stot
            if out:
                print('Rgate[{:.3f}] on mode {}'.format(phis[i][0], phis[i][1]) )

    #Apply the symplectic to the state
    state.apply_symplectic(Stot)
                

    #Apply displacements at the end of the circuit if any
    alphas = params['alpha']
    

    if alphas:
        disp = np.zeros(2*nmodes)
        for i in range(len(alphas)):
            alpha = alpha[i][0]
            #disp += xxpp_to_xpxp(expand_vector(alpha, alpha[i][1])) #hbar = 2 by default
            disp += expand_vector(alpha, alpha[i][1]) #hbar = 2 by default
            if out:
                print(r'Dgate[{:.3f}] on mode {}'.format(alpha, alphas[i][1]))

        state.apply_displacement(disp)
            
    #Pure/thermal losses,
    loss = params['loss']
    if loss:
        etas = np.ones(nmodes)
        nbars = np.zeros(nmodes)
        
        for i in range(len(loss)):
            idx = loss[i][2]
            etas[idx] = loss[i][0]
            nbars[idx] = loss[i][1]
            
            if out:
                print(r'{:.1f}% loss channel with {:.1f}photons on mode {}'.format(loss[i][0]*100,loss[i][1],idx)) 
                
        #Apply the losses       
        state.apply_loss(etas, nbars) 
    
        
    return state