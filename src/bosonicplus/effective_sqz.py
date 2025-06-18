import numpy as np
from bosonicplus.charfun import char_fun
from bosonicplus.conversions import dB_to_r, Delta_to_dB


def get_gkp_stabilizer(lattice):

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
    

def effective_sqz(state, lattice : str):
    """Get the effective squeezing of a state according to the Terhal definition.

    Args:
        state : base.State object
        lattice: The GKP lattice and direction, sx, sp, rx, rp, hx, hp
    """

    alpha = get_gkp_stabilizer(lattice)

    
    #Symmetrically
    f1 = char_fun(state, alpha)
    f2 = char_fun(state, -alpha)

    
    D1 = np.sqrt(-2/np.abs(alpha)**2*np.log(np.abs(f1)))
    D2 = np.sqrt(-2/np.abs(alpha)**2*np.log(np.abs(f2)))
    
    Delta = 0.5 * (D1+D2)
    
    return float(Delta)





