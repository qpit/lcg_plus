import numpy as np
from bosonicplus.conversions import dB_to_r

def gen_Clements_indices(nmodes):
    """ Get a list of the beamsplitter indices of a Clements interferometer.
    From https://github.com/XanaduAI/approximate-GKP-prep/blob/master/StatePrepCircuits.ipynb
    
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
    Generate random interferometer parameters for interferometer in Clements, cascade or inv_cascade convension
    
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
    sqz = list(zip(rs, rs_angle, range(nmodes)))

    # Beamsplitters
    if bs_arrange == 'Clements':
        inds = gen_Clements_indices(nmodes)
    elif bs_arrange == 'cascade':
        inds = gen_cascade_indices(nmodes)
    elif bs_arrange == 'inv_cascade':
        inds = gen_inv_cascade_indices(nmodes)
    else:
        raise ValueError('bs_arrange must be either str(Clements), str(cascade) or str(inv_cascade).')
        
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
    sqz = list(zip(rs, rs_angle, range(nmodes)))

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
