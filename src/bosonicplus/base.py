import numpy as np
from thewalrus.symplectic import xpxp_to_xxpp, xxpp_to_xpxp, expand, rotation
from bosonicplus.operations.measurements import project_fock_coherent, project_ppnrd_thermal, project_homodyne
from bosonicplus.states.wigner import Gauss
from bosonicplus.from_sf import chop_in_blocks_multi, chop_in_blocks_vector_multi
import itertools as it
from scipy.linalg import block_diag
from mpmath import mp

hbar = 2

class State:
    """Store the wigner function of a state as a linear combination of Gaussians
    by tracking the means, covs, and weights of the Gaussians
    """
    def __init__(self, num_modes = 1):
        """Initialise in vacuum by default
        """
        self.num_modes = num_modes
        self.means = np.array([np.zeros(self.num_modes * 2)])
        self.covs = np.array([np.eye(self.num_modes * 2)]) * hbar / 2 #hbar = 2
        self.weights = np.array([1])
        #self.data = [self.means, self.covs, self.weights]
        self.num_weights = len(self.weights)
        self.num_covs = len(self.covs) #Relevant for faster calculations
        self.ordering = 'xpxp'
        self.probability = 1 #For measurements and normalisation

    def update_data(self, new_data : tuple, ordering = 'xpxp'):
        """Insert a custom data tuple, new_data = [means, covs, weights]. 
        This overrides the existing state data completely.
        
        To do: what if new_data is in a different ordering?
        To do: fix weird weight change by writing separate update covs and update means methods
        """
        #self.data = new_data
        self.means = new_data[0]
        self.covs = new_data[1]
        self.weights = new_data[2]
        self.num_weights = len(self.weights)
        self.ordering = ordering
        self.probability = np.sum(self.weights) 
        
        self.num_modes = int(np.shape(self.means)[-1]/2)
        if len(self.covs.shape) != 3: 
            self.covs = np.array([self.covs]) #Quick fix for places where covs is (2,2), not (1,2,2)
        self.num_covs = len(self.covs)
    def get_mean_photons(self):
        """https://github.com/XanaduAI/thewalrus/blob/master/thewalrus/symplectic.py line 354,
        but this only worked for a single mode covariance matrix. Here its adapted to get the
        sum of average photon numbers for multimode states.
        """
        cov_tr = np.trace(self.covs, axis1=1,axis2=2)
        mu_sq = np.einsum("...j,...j", self.means,self.means)

        covsq_tr = np.trace(np.einsum("...jk,...kl", self.covs, self.covs), axis1 = 1, axis2=2)
        mucov = np.einsum("...j,...jk,...k", self.means, self.covs, self.means)
        
        num = self.num_modes
            
        ex = np.sum(self.weights*((cov_tr + mu_sq)/(2*hbar)-1/2*num)/self.probability)
        var = np.sum(self.weights*((covsq_tr+2*mucov)/(2 * hbar**2) - 1 / 4*num)/self.probability)
        #Variance might not be compatible with the sum-of-gaussians framework
        
        return ex, var

    def to_xpxp(self):
        """Change the ordering from xpxp to xxpp
        """
        if self.ordering == 'xpxp':
            raise ValueError('Already in xpxp ordering.')
        else:
            means, covs, weights = self.data
            means = np.array([xxpp_to_xpxp(i) for i in means])
            covs = np.array([xxpp_to_xpxp(i) for i in covs]) #quick workaround
            self.update_data([means, covs, weights], 'xpxp')
            #self.ordering = 'xpxp'

    def to_xxpp(self):
        """Change the ordering from xpxp to xxpp
        """
        if self.ordering == 'xxpp':
            raise ValueError('Already in xxpp ordering.')
        else:
            means, covs, weights = self.data
            means = np.array([xpxp_to_xxpp(i) for i in means])
            covs = np.array([xpxp_to_xxpp(i) for i in covs]) #quick workaround
            self.update_data([means, covs, weights], 'xxpp')
            #self.ordering = 'xxpp'

    def apply_symplectic_fast(self, S, modes, ordering = 'xpxp'):
        """Partition total system into A and B modes. Act with symplectic on just the B modes, and
        reassemble the covariance matrix and disp vector from the updated elements. 
        This method has better performance than apply_symplectic() when the number of modes is large (> 20). 
        """

        if len(modes) != S.shape[0]/2:
            raise ValueError('Symplectic must have same dimension as the modes list')
        if self.num_modes == 2 and len(modes)==2:
            raise ValueError('Use apply_symplectic.')
        if ordering != self.ordering:
            raise ValueError('Symplectic not in same ordering as state.')
        
        mode_ind = np.concatenate((2 * np.array(modes), 2 * np.array(modes) + 1)) #in xxpp 
        mode_ind = xxpp_to_xpxp(mode_ind) #back to xpxp
        
        mode_inds = np.arange(2*self.num_modes)
        
        mode_ind_rest = list(set(mode_inds) - set(mode_ind))
        
        A, AB, B = chop_in_blocks_multi(self.covs, mode_ind) #Chop the system into A modes: untouched, and B modes: the modes the symplectic is applied to
        a, b = chop_in_blocks_vector_multi(self.means, mode_ind)
        
        Bnew = np.einsum("...jk,...kl,...lm",S,B,S.T)
        ABnew = np.einsum("...jk,...kl",AB,S.T)
        bnew = np.einsum("...jk,...k",S,b)
        
        nw = self.num_weights #Number of weights
        if self.covs.shape[0] != 1:
            self.covs[np.ix_(np.arange(nw),mode_ind,mode_ind)] = Bnew 
            self.covs[np.ix_(np.arange(nw),mode_ind_rest, mode_ind)] = ABnew
            self.covs[np.ix_(np.arange(nw),mode_ind, mode_ind_rest)] = np.transpose(ABnew, axes = [0,2,1])
        else:
            self.covs[np.ix_([0],mode_ind,mode_ind)] = Bnew 
            self.covs[np.ix_([0],mode_ind_rest, mode_ind)] = ABnew
            self.covs[np.ix_([0],mode_ind, mode_ind_rest)] = np.transpose(ABnew, axes = [0,2,1])
        
        self.means[np.ix_(np.arange(nw),mode_ind)] = bnew

    
    def apply_symplectic(self, S : np.ndarray, ordering = 'xpxp'):
        """ Apply symplectic to data
            S (array) : symplectic matrix
            ordering (str): ordering of S

            To do: check covs update when num_covs != 1

            Note that symplectics from thewalrus are in xxpp ordering
        """
        means = self.means
        covs = self.covs
        
        #First check that S has the correct dimensions
        if np.shape(S)[0] != int(np.shape(means)[-1]):
            raise ValueError('S must must be 2nmodes x 2nmodes. ')
        #Bug
        #if ordering != self.ordering:
            #raise ValueError('Symplectic not in same ordering as state.')
            
        if self.num_covs == 1: 
            new_covs = S @ covs @ S.T
        else:
            #Didn't test if this works
            new_covs = np.einsum("...jk,...kl,...lm", S[np.newaxis,:], covs, (S.T)[np.newaxis, :])

        new_means = np.einsum("...jk,...k", S[np.newaxis,:], means)
    
        #Update data
        self.means = new_means
        self.covs = new_covs
        #self.update_data([new_means, new_covs, weights])
        
        
    def apply_displacement(self, d: np.ndarray, ordering = 'xpxp'):
        r""" d must be in the same ordering (xxpp) or (xpxp)
        To do: add test of d shape compatibility
        """
        
        if len(d) != self.means.shape[-1]:
            raise ValueError('d must be 2 x nmodes.')
        if ordering != self.ordering:
            raise ValueError('Ordering of d must be the same as ordering of means.')
        #Update data
        self.means += d
        
    

    def apply_loss(self, etas, nbars):
        """Apply loss to (multimode) state 

        To do: add etas and nbars shape check, add case where num covs !=1
    
        Gaussian state undergo a attenuation channel in the following way:
            cov = X @ cov @ X.T + Y
            means = X @ means

            where X = sqrt(eta)*I, Y = (1-eta) * hbar / 2 * (2*nbar+1) * I
    
        Args:
            etas (array): array giving transmittivity of each mode
            nbars (array): array giving number of photons in environment each mode is coupled to
    
        Returns:
           updates means, covs
        """
        num_modes = self.num_modes
        means = self.means
        cov = self.covs

        if self.ordering == 'xxpp':
            X = np.diag(np.repeat(np.sqrt(etas),2))
            Y = np.diag(np.repeat( (1-etas) * hbar / 2 * (2*nbars + 1) ,2 ))
        elif self.ordering == 'xpxp':
            X = xxpp_to_xpxp(np.diag(np.repeat(np.sqrt(etas),2)))
            Y = xxpp_to_xpxp(np.diag(np.repeat( (1-etas) * hbar / 2 * (2*nbars + 1) ,2 )))
        
        means = np.einsum("...jk,...k", X, means)
        cov = X @ cov @ X.T
        cov += Y
        
        #Update data
        self.means = means
        self.covs = cov

    def apply_gain(self, Gs):
        """Apply gain to (multimode) state

        To do: add etas and nbars shape check, add case where num covs !=1
    
        Gaussian state undergo an amplification channel in the following way:
            cov = X @ cov @ X.T + Y
            means = X @ means

            X = sqrt(G)*I, Y = (G-1) * hbar / 2 * I
    
        Args:
            Gs (array): array giving the gain in each mode

        Returns:
            updates means, covs
           
        """
        num_modes = self.num_modes
        means = self.means
        cov = self.covs

        if self.ordering == 'xxpp':
            X = np.diag(np.repeat(np.sqrt(Gs),2))
            Y = np.diag(np.repeat( (Gs-1) * hbar / 2 ,2 ))
        elif self.ordering == 'xpxp':
            X = xxpp_to_xpxp(np.diag(np.repeat(np.sqrt(Gs),2)))
            Y = xxpp_to_xpxp(np.diag(np.repeat( (Gs-1) * hbar / 2 ,2 )))
        
        means = np.einsum("...jk,...k", X, means)
        cov = X @ cov @ X.T
        cov += Y
        
        #Update data
        self.means = means
        self.covs = cov

    def post_select_fock_coherent(self, mode, n, inf = 1e-4, out = False):
        """Post select on counting n photons in mode. New state has one less mode, so be careful with indexing.
    
        Args: 
            mode (int) : measured mode index 
            n (int) : photon number
    
        Returns: updates data
        """
        #Make sure that ordering is xpxp first
        if self.ordering != 'xpxp':
            self.to_xpxp()

        data_in = self.means, self.covs, self.weights
            
        data_out, prob = project_fock_coherent(n, data_in, mode, inf)
    
        if out:
            print(f'Measuring {n} photons in mode {mode}.')
            print(f'Data shape before measurement, {[i.shape for i in data_in]}.')
            print('Probability of measurement = {:.3e}'.format(prob))
            print(f'Data shape after measurement, {[i.shape for i in data_out]}')

        self.update_data(data_out)
        self.probability = prob


    def post_select_ppnrd_thermal(self, mode, n, M, out =False):
        """
        Detect mode wth pPNRD registering n clicks by demultiplexing into M on/off detectors.
        The pPNRD POVM is written as a linear combination of Gaussians (thermal states) and the
        circuit's Gaussian means, covs and weights are updated according to the Gaussian transformation rules of 
        Bourassa et al. 10.1103/PRXQuantum.2.040315 . 
    
        Extension/generalisation of code from strawberryfield's bosonicbackend. 
        
        To do: 
            Write down formula in documentation.
        
        Args: 
            mode (int): mode to be detected
            n (int): number of clicks detected
            M (int): number of on/off detectors in the click-detector    
            out (bool): print output text
            
        Returns: updates circuit object
        
        """
        if n > M:
            raise ValueError('Number of clicks cannot exceed click detectors.')
        #Make sure that ordering is xpxp first
        if self.ordering != 'xpxp':
            self.to_xpxp()

        data_in = self.means, self.covs, self.weights

        data_out, prob = project_ppnrd_thermal(data_in, mode, n, M)
    
        if out:
            print(f'Measuring {n} clicks in mode {mode}.')
            print(f'Data shape before measurement, {[i.shape for i in data_in]}.')
            print('Probability of measurement = {:.3e}'.format(prob))
            print(f'Data shape after measurement, {[i.shape for i in data_out]}')
            
        self.update_data(data_out)
        self.probability = prob


    def post_select_homodyne(self, mode, angle, result, MP = False):

        #First, rotate the mode by -angle
        S = xxpp_to_xpxp(expand(rotation(-angle), mode, self.num_modes))
        self.apply_symplectic(S)

        data_in = self.means, self.covs, self.weights
        
        data_out, prob = project_homodyne(data_in, mode, result, MP)
        self.update_data(data_out)
        #self.probability *= prob  
        self.probability = prob 
        

    def get_wigner(self, x = None, p = None, indices = None, MP = False):
        """
        Obtain the (single mode) Wigner function on a grid of phase space points
        """
        if x is None:
            x = np.linspace(-10,10,200)
        if p is None:
            p = x

        if indices is not None: 
            sigmaA, sigmaB, sigmaAB = chop_in_blocks_multi(self.covs, indices)
            
            muA, muB = chop_in_blocks_vector_multi(self.means, indices)
           
            W = 0
            if len(sigmaA) == 1:
                for i, mu in enumerate(muA):
                    W += self.weights[i] * Gauss(sigmaA, mu, x, p, MP)/self.probability
            else:
                for i, mu in enumerate(muA):
                    W += self.weights[i] * Gauss(sigmaA[i], mu, x, p, MP)/self.probability
        else: 
            if self.num_modes != 1:
                raise ValueError('State has multiple modes, please specify indices.')
            W = 0
            if len(self.covs) == 1:
                for i, mu in enumerate(self.means):
                    W += self.weights[i] * Gauss(self.covs, mu, x, p, MP)/self.probability
            else:
                for i, mu in enumerate(self.means):
                    W += self.weights[i] * Gauss(self.covs[i], mu, x, p, MP)/self.probability
        return W

    def multimode_copy(self, n):
        """Duplicate a single mode state onto n modes.
        """
        
        # Check number of modes in state
        if self.num_modes != 1:
            raise ValueError('This is a multimode state. Can only copy make copies of single mode states.')
            
        means, cov, weights = self.means, self.covs, self.weights
        print('input data shape', means.shape, cov.shape, weights.shape)
        
        if self.num_covs == 1:
            new_cov = np.array([block_diag(*np.repeat(cov,n,axis = 0))])
    
        new_weights = np.prod(np.array(list(it.product(weights.tolist(), repeat = n, ))), axis = 1)
        new_means = np.reshape(np.array(list(it.product(means, repeat = n))), (len(weights)**n, n*2) )
        #new_cov = np.array([block_diag(*list(it.product(np.squeeze(cov), repeat = n)))])
    
        print('new data shape', new_means.shape, new_cov.shape, new_weights.shape)
        
        data_new = new_means, new_cov, new_weights
        
        self.update_data(data_new)

    def add_state(self, state, MP=False):
        """Tensor product of a state with a user-specified state in sum of Gaussian representation
        """
        N = self.num_modes
        M = state.num_modes
        
        means1, cov1, weights1 = self.means, self.covs, self.weights
        means2, cov2, weights2 = state.means, state.covs, state.weights
        
        #In coherent picture, covariances are the same for every weight
        if len(cov1) != 1 or len(cov2) != 1:
            new_cov = np.array([block_diag(*i) for i in list(it.product(cov1,cov2))])
        else:
            
            new_cov = np.array([block_diag(*list([np.squeeze(cov1),np.squeeze(cov2)]))])
        
        if MP:
            new_weights = np.array([mp.fprod(i) for i in np.array(list(it.product(weights1, weights2)))])
        else:
            new_weights = np.prod(np.array(list(it.product(weights1, weights2))),axis=1)
        
        K = len(new_weights)
        new_means = np.array(list(it.product(means1,means2))).reshape((K,2*(N+M)))
        
        data_new = new_means, new_cov, new_weights
        
        self.update_data(data_new)
    
    
    