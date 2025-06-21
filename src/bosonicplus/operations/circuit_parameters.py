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
        

def gen_interferometer_params(nmodes, r_max_dB, bs_arrange = 'Clements', setting = 'no_phase'):
    """
    Generate random interferometer parameters for interferometer in Clements, cascade or inv_cascade convension
    
    Args: 
        nmodes (int): number of modes
        r_max_dB (float): max squeezing in dB
        bs_arrange (str): beamsplitter arrangement, Clements, cascade, inv_cascade
        setting (str): circuit setting, either no_phase or two_mode_squeezing
    Returns:
        params (dict): dictionary
    """
  
    # Beamsplitters/two-mode squeezers
    if bs_arrange == 'Clements':
        inds = gen_Clements_indices(nmodes)
    elif bs_arrange == 'cascade':
        inds = gen_cascade_indices(nmodes)
    elif bs_arrange == 'inv_cascade':
        inds = gen_inv_cascade_indices(nmodes)

    else:
        raise ValueError('bs_arrange must be either str(Clements), str(cascade), str(inv_cascade)')
        
    nbs = len(inds)

    if setting == 'no_phase':
        r = dB_to_r(r_max_dB)
        rs = np.random.uniform(-r,r,nmodes)
        rs_angle = np.zeros(nmodes) #Ignore the squeezing angle.
        sqz = list(zip(rs, rs_angle))
        
        thetas = np.random.uniform(0.1,np.pi/2-0.1,nbs)
        phis = np.zeros(nbs) #Ignore the beam splitter phases
 
        bs = list(zip(thetas, phis, inds))

        phis_extra = [] #No extra rotation
        alpha = [] #No displacement
        
    elif setting == 'two_mode_squeezing':
        sqz = []
        r = dB_to_r(r_max_dB)
        rs = np.random.uniform(0,r,nmodes)
        phis = np.random.uniform(-np.pi,np.pi, nmodes)
        bs = list(zip(rs, phis, inds))
        phis_extra = [] #No extra rotation
        alpha = [] #No displacement
        
    # Extra phases
    #if phases:
     #   phis = np.random.uniform(-np.pi,np.pi,nmodes)
      #  phis_extra = list(phis)
    #else:
     #   phis_extra = []

    #Displacement
    #if disp:
     #   alphas = np.random.uniform(0,disp,nmodes)
        #alphas_phi = np.random.uniform(-np.pi,np.pi,nmodes)
      #  alphas_phi = np.zeros(nmodes) #Ignore displacement phase
       # alpha = list(zip(alphas, alphas_phi))
    #else:
     #   alpha = []


    params = {'sqz': sqz, 'bs': bs, 'phis':  phis_extra, 'alphas': alpha}
    
    return params

def params_to_1D_array(params_dict, setting = 'no_phase'):
    """Unpack dict into 1D array of params in a specific order to use as arg in scipy.optimize.minimize/basinhopping

    Returns: 
        
    """
    sqz = params_dict['sqz']
    bs = params_dict['bs']
    phis_extra = params_dict['phis']
    disp = params_dict['alphas']
    
    if sqz:
        rs, rs_phi = zip(*params_dict['sqz'])
        
    if disp:
        alpha, alpha_phi = zip(*params_dict['alphas'])
        

    bs_thetas, bs_phis, inds = zip(*params_dict['bs']) #BS indices are inferred in circuit building function

    
    if setting == 'no_phase':
        #Ignore ALL phases
        alpha = [] #No displacement
        params = [rs, bs_thetas, alpha]
    if setting == 'two_mode_squeezing':
        params = zip(bs_thetas, bs_phis) #Concantenate pairwise
        
    return [i for row in params for i in row] #Flatten manually
    

def unpack_params(params, nmodes, bs_arrange = 'Clements', setting = 'no_phase'):
    
    """
    Unpack squeezing, beasmplitter transmittivity and phase, and extra rotation parameters from a 1D parameter array.

    Total available parameters:
    params = [rs, rphis, thetas, phis, phis_extra, alpha, alpha_phis] 

    no_phase setting: 
    params = [rs, thetas, alpha]
    
    Args:
        params (ndarray): 
        nmodes (int): number of modes in the circuit
    Returns:
        (tuple): rs, thetas, phis, phis_extra
    """
    if bs_arrange == 'Clements':
        nbs = int(nmodes*(nmodes-1)/2)
    elif bs_arrange == 'cascade':
        nbs = int(nmodes - 1)
    elif bs_arrange == 'inv_cascade':
        nbs = int(nmodes - 1)

    if setting == 'no_phase':
        
        rs = params[0:nmodes]
        thetas = params[nmodes: nmodes + nbs]
        alphas = params[nmodes+nbs:]
        return rs, thetas, alphas
        
    elif setting == 'two_mode_squeezing':
        rs = params[0 : nbs]
        phis = params[nbs:]
        alphas = np.array([])
        return rs, phis, alphas
        
    else:
        raise ValueError('This setting is not implemented')
        
def params_to_dict(params, nmodes, bs_arrange = 'Clements', setting = 'no_phase'):
    """
    Get a dictionary over the parameters arranged in a 1D list.
    """

    if setting == 'no_phase':
        
        phis_sq = np.zeros(nmodes)
        phis_extra = []
        rs, thetas, alpha = unpack_params(params, nmodes, bs_arrange, setting)
        sqz = list(zip(rs, phis_sq)) #Add zero angle to squeezing
        phis_bs = np.zeros(len(thetas))
        
        
    elif setting == 'two_mode_squeezing':
        thetas, phis_bs, alpha = unpack_params(params, nmodes, bs_arrange, setting)
        sqz = []
        phis_extra = []
    else:
        raise ValueError('setting isnt implemented.')

        
    if len(alpha) != 0 : 
        alpha = list(zip(alpha, phis))
     
    # Beamsplitters
    if bs_arrange == 'Clements':
        bs = list(zip(thetas, phis_bs, gen_Clements_indices(nmodes)))
    elif bs_arrange =='cascade':
        bs = list(zip(thetas, phis_bs, gen_cascade_indices(nmodes)))
    elif bs_arrange =='inv_cascade':
        bs = list(zip(thetas, phis_bs, gen_inv_cascade_indices(nmodes)))
        
   
    return {'sqz': sqz, 'bs': bs, 'phis': phis_extra, 'alphas': alpha}

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
