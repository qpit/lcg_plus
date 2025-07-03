import numpy as np
from thewalrus.symplectic import xpxp_to_xxpp, xxpp_to_xpxp, expand, rotation
from thewalrus.decompositions import williamson
from bosonicplus.operations.measurements import project_fock_coherent, project_ppnrd_thermal, project_homodyne, project_fock_thermal, project_fock_coherent_gradients
from bosonicplus.states.wigner import Gauss
from bosonicplus.states.coherent import gen_fock_superpos_coherent, get_cnm, eps_superpos_coherent
from bosonicplus.states.reduce import reduce, reduce_log, reduce_log_full, reduce_log_pure

from bosonicplus.sampling import *

from bosonicplus.from_sf import chop_in_blocks_multi, chop_in_blocks_vector_multi
import itertools as it
from scipy.linalg import block_diag
from scipy.special import logsumexp
from mpmath import mp
from math import fsum



hbar = 2

class State:
    """Store the wigner function of a state as a linear combination of Gaussians
    by tracking the means, covs, and weights of the Gaussians

    
    """
    def __init__(self, num_modes = 1, hbar = 2):
        """Initialise in vacuum by default
        """
        self.hbar = hbar
        self.num_modes = num_modes
        self.means = np.array([np.zeros(self.num_modes * 2)])
        self.covs = np.array([np.eye(self.num_modes * 2)]) * self.hbar / 2
        self.log_weights = np.array([0]) #log_weights
        self.weights = np.array([1]) # weights
        self.num_weights = len(self.weights)
        self.num_covs = len(self.covs) #Relevant for faster calculations
        self.num_k = self.num_weights #Number of Gaussian to be treated "normally" - upper triangular form.
        self.ordering = 'xpxp' #xpxp by default.
        self.norm = 1 #Normalisation (relevant when doing measurements)

    def update_data(self, new_data : tuple, ordering = 'xpxp'):
        """Insert a custom data tuple, new_data = [means, covs, log_weights, k]. 
        This overrides the existing state data completely.
        
        To do: what if new_data is in a different ordering?
        """
        
        if len(new_data) != 4:
            raise ValueError('new_data must be [means, covs, log_weights, k] tuple.')
            
        self.means, self.covs, self.log_weights, self.num_k = new_data

        self.num_weights = len(self.log_weights)
        
        self.ordering = ordering
        
        self.num_modes = int(np.shape(self.means)[-1]/2)
        
        if len(self.covs.shape) != 3: 
            self.covs = np.array([self.covs]) #Quick fix for places where covs is shape (2,2), not (1,2,2)
            
        self.num_covs = len(self.covs)
        self.weights = np.exp(self.log_weights)

    def update_gradients(self, new_gradients : tuple):
        """Insert a custom gradient tuple. Overrides any existing gradient data.
        """
        
        self.means_partial, self.covs_partial, self.log_weights_partial = new_gradients
        

    def get_norm(self):
        r"""Calculate the norm by adding the weights together
        """
         
        log_norm = logsumexp(self.log_weights)

        if self.num_k != self.num_weights:
            self.log_norm = log_norm
            self.norm = np.exp(log_norm).real

        else:   
            self.log_norm = log_norm
            self.norm = np.real_if_close(np.exp(log_norm))

    def normalise(self):
        r"""Subtract norm from log_weights
        """
        if self.num_k != self.num_weights:
            self.log_weights -= np.log(self.norm) 
        else:
            self.log_weights -= self.log_norm 
            
        self.weights /= self.norm

        self.get_norm() #Populate norm
              
    def get_photon_number_moments(self):
        """Dodonov and Man'ko https://doi.org/10.1103/PhysRevA.50.813
        """

        #Get rid of the hbar 
        covs = self.covs / self.hbar
        means = self.means /np.sqrt(self.hbar)
        
        cov_tr = np.trace(covs, axis1=1,axis2=2)
        cov_det = np.linalg.det(covs)
        

        mu_sq = np.einsum("...j,...j", means, means)

        exk = 1/2 *(cov_tr + mu_sq - 1)

        ex = np.exp(logsumexp(self.log_weights + np.log(exk))) / self.norm

        #covsq_tr = np.trace(np.einsum("...jk,...kl", covs, covs), axis1 = 1, axis2=2)
        mucov = np.einsum("...j,...jk,...k", means, covs, means)

        vark = 1/2 * (cov_tr**2 -2 * cov_det - 0.5) + mucov
    
        var = np.exp(logsumexp(self.log_weights + np.log(vark))) /self.norm 

        #num = self.num_modes
            
        #ex = np.sum(self.weights*((cov_tr + mu_sq)/(2*self.hbar)-1/2*num))/self.norm
        
        #var = np.sum(self.weights*((cov_tr**2-2*np.linalg.det(self.covs)+2*mucov)/(2 * self.hbar**2) - 1 / 4*num)/self.norm)
       
        
        return ex, var

    def get_mean(self, MP = False):
        """Get first moment
        """
        if MP:
            dim = len(self.means[0])
            mu = np.array([float(mp.re(mp.fsum(self.weights * self.means[:,i]))) for i in range(dim)])
        else:
            mu = np.real_if_close(np.sum(self.weights[:,np.newaxis] * self.means, axis = 0)) 
        
        if self.num_k != self.num_weights:
            mu = mu.real
        
        self.mean = mu #Set the first moment
        return mu
                

    def get_cov(self, MP = False):
        """Get second moment. 
        """
        offset = np.tensordot(self.mean, self.mean, axes =0)
        
        sigma_tilde = self.covs + np.einsum("...j,...k", self.means, self.means)
        
        if MP:
            dim = len(self.covs[0][0])
            cov = np.array([[float(mp.re(mp.fsum(self.weights
                                                 * sigma_tilde[:,i,j]))) for i in range(dim)] for j in range(dim)])
        else:  
            cov = np.sum(self.weights[:,np.newaxis, np.newaxis] * sigma_tilde ,axis =0)
            
        
        sigma = np.real_if_close(cov - offset)
     
            
        self.sigma = sigma #Set the 2nd moment
        return sigma

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

        if self.ordering == 'xpxp':
            X = np.diag(np.repeat(np.sqrt(etas),2))
            Y = np.diag(np.repeat( (1-etas) * self.hbar / 2 * (2*nbars + 1) ,2 ))
        elif self.ordering == 'xxpp':
            X = xpxp_to_xxpp(np.diag(np.repeat(np.sqrt(etas),2)))
            Y = xpxp_to_xxpp(np.diag(np.repeat( (1-etas) * self.hbar / 2 * (2*nbars + 1) ,2 )))

        means = np.einsum("...jk,...k", X, means)
        cov = X @ cov @ X.T
        cov += Y
        
        #Update data
        self.means = means
        self.covs = cov

        #Update the gradients if any
        if hasattr(self, "means_partial"):
            self.means_partial =np.einsum("...jk,...k", X, self.means_partial)
            self.covs_partial =np.einsum("...jk,...kl,...lm", X, self.covs_partial, X.T)

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

        if self.ordering == 'xpxp':
            X = np.diag(np.repeat(np.sqrt(Gs),2))
            Y = np.diag(np.repeat( (Gs-1) * self.hbar / 2 ,2 ))
        elif self.ordering == 'xxpp':
            X = xxpp_to_xpxp(np.diag(np.repeat(np.sqrt(Gs),2)))
            Y = xxpp_to_xpxp(np.diag(np.repeat( (Gs-1) * self.hbar / 2 ,2 )))
        
        means = np.einsum("...jk,...k", X, means)
        cov = X @ cov @ X.T
        cov += Y
        
        #Update data
        self.means = means
        self.covs = cov

        #Update the gradients if any
        if hasattr(self, "means_partial"):
            self.means_partial =np.einsum("...jk,...k", X, self.means_partial)
            self.covs_partial =np.einsum("...jk,...kl,...lm", X, self.covs_partial, X.T)

    def post_select_fock_coherent(self, mode, n, inf = 1e-4, red_gauss = True, out = False):
        """Post select on counting n photons in mode. New state has one less mode, so be careful with indexing.
    
        Args: 
            mode (int) : measured mode index 
            n (int) : photon number
    
        Returns: updates data
        """
        #Make sure that ordering is xpxp first
        if self.ordering != 'xpxp':
            self.to_xpxp()

        data_in = self.means, self.covs, self.log_weights
        
        if red_gauss:
            data_out = project_fock_coherent(n, data_in, mode, inf, self.num_k)
            
        else: 
            data_out = project_fock_coherent(n, data_in, mode, inf)
        
        self.update_data(data_out)
        self.get_norm()
    
        if out:
            print(f'Measuring {n} photons in mode {mode}.')
            print(f'Data shape before measurement, {[i.shape for i in data_in[0:2]]}.')
            print('Probability of measurement = {:.3e}'.format(self.norm))
            print(f'Data shape after measurement, {[i.shape for i in data_out[0:2]]}')


    def post_select_fock_coherent_gradients(self, mode, n, inf = 1e-4, out = False):
        """Post select on counting n photons in given mode. New state has one less mode, so be careful with indexing.
    
        Args: 
            mode (int) : measured mode index 
            n (int) : photon number
    
        Returns: updates data
        """
        #Make sure that ordering is xpxp first
        if self.ordering != 'xpxp':
            self.to_xpxp()

        data_in = self.means, self.covs, self.log_weights, self.num_k
        data_partial = self.means_partial, self.covs_partial, self.log_weights_partial

        
        data_out, data_gradients = project_fock_coherent_gradients(n, data_in, data_partial, mode, inf)
        
        self.update_data(data_out)
        self.update_gradients(data_gradients)
        
        self.get_norm()
    
        if out:
            print(f'Measuring {n} photons in mode {mode}.')
            print(f'Data shape before measurement, {[i.shape for i in data_in[0:2]]}.')
            print('Probability of measurement = {:.3e}'.format(self.norm))
            print(f'Data shape after measurement, {[i.shape for i in data_out[0:2]]}')

    def post_select_fock_thermal(self, mode, n, r =0.05, out = False):
        #Make sure that ordering is xpxp first
        if self.ordering != 'xpxp':
            self.to_xpxp()

        data_in = self.means, self.covs, self.log_weights
            
    
        data_out = project_fock_thermal(data_in, mode, n, r)
        self.update_data(data_out)
    
        if out:
            print(f'Measuring {n} photons in mode {mode}.')
            print(f'Data shape before measurement, {[i.shape for i in data_in[0:2]]}.')
            print('Probability of measurement = {:.3e}'.format(self.norm))
            print(f'Data shape after measurement, {[i.shape for i in data_out[0:2]]}')


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
        if self.num_k != self.num_weights:
            raise ValueError('This measurement is not yet compatible with fast gaussian rep.')
        #if n ==M:
            #raise ValueError('Current trace implementation is questionable.')
        #Make sure that ordering is xpxp first
        if self.ordering != 'xpxp':
            self.to_xpxp()

        data_in = self.means, self.covs, self.log_weights

        data_out = project_ppnrd_thermal(data_in, mode, n, M)

        self.update_data(data_out)
        self.get_norm()
    
        if out:
            print(f'Measuring {n} clicks in mode {mode}.')
            print(f'Data shape before measurement, {[i.shape for i in data_in[0:2]]}.')
            print('Probability of measurement = {:.3e}'.format(self.norm))
            print(f'Data shape after measurement, {[i.shape for i in data_out[0:2]]}')
            
        
       


    def post_select_homodyne(self, mode, angle, result):

        #First, rotate the mode by -angle
        S = xxpp_to_xpxp(expand(rotation(-angle), mode, self.num_modes))
        self.apply_symplectic(S)

        data_in = self.means, self.covs, self.log_weights
        
        data_out = project_homodyne(data_in, mode, result, self.num_k)
        self.update_data(data_out)
        self.get_norm()
        

    def get_wigner_bosonic(self, xvec, pvec, indices = None):
        """Adapted from strawberryfields.backends.states for BaseBosonicState
        """

        if indices is not None: 
            sigmaA, sigmaAB, covs = chop_in_blocks_multi(self.covs, indices)
            muA, means = chop_in_blocks_vector_multi(self.means, indices)
        else:
            if self.num_modes != 1:
                raise ValueError('State has multiple modes, please specify indices.')
            means = self.means
            covs = self.covs
            
        weights = self.weights
        norm = self.norm

        
        
        X, P = np.meshgrid(xvec, -pvec, sparse=True) #Use -pvec because of matplotlib.imshow y axis convention. Can cause issues if comparing with analytical Wigner functions..
        
        wigner = 0
        for i, weight_i in enumerate(weights):
        
            if X.shape == P.shape:
                arr = np.array([X - means[i, 0], P - means[i, 1]])
                arr = arr.squeeze()
                
            else:
                # need to specify dtype for creating an ndarray from ragged
                # sequences
                arr = np.array([X - means[i, 0], P - means[i, 1]], dtype=object)

            if len(covs) ==1:
                exp_arg = arr @ np.linalg.inv(covs[0]) @ arr
                prefactor = 1 / (np.sqrt(np.linalg.det(2 * np.pi * covs[0])))
            else: 
                exp_arg = arr @ np.linalg.inv(covs[i]) @ arr
                prefactor = 1 / (np.sqrt(np.linalg.det(2 * np.pi * covs[i])))

        
        
            wigner += (weight_i * prefactor) * np.exp(-0.5 * (exp_arg))
        return np.real_if_close(wigner/norm)

    def get_wigner_log(self, x, p, indices = None):
        
        if indices is not None: 
            sigmaA, sigmaAB, covs = chop_in_blocks_multi(self.covs, indices)
            muA, means = chop_in_blocks_vector_multi(self.means, indices)
        else:
            if self.num_modes != 1:
                raise ValueError('State has multiple modes, please specify indices.')
            means = self.means
            covs = self.covs
    
        log_weights = self.log_weights
        #log_norm = self.log_norm
        norm = self.norm
        means = self.means
        covs = self.covs
        
        X, P = np.meshgrid(x, -p, sparse=False) #Use -p because of matplotlib.imshow y axis convention. Can cause issues if comparing with analytical Wigner functions..
        
        Q = np.array([X,P])
            
        arr = Q[np.newaxis,:] - means[:,:, np.newaxis,np.newaxis]
        
        if len(covs) ==1:
           
            arr=np.transpose(arr, [2,3,0,1])
            
            exp_arg = -0.5 * np.einsum("...j,...jk,...k", arr, np.linalg.inv(covs[0])[np.newaxis,np.newaxis,:,:], arr)
            
            prefactor = 1 / (np.sqrt(np.linalg.det(2 * np.pi * covs[0])))
            
        
        else:
            #raise ValueError('Currently not working when num_covs != 1.')
            arr=np.transpose(arr, [2,3,0,1])
            
            exp_arg = -0.5 * np.einsum("...j,...jk,...k", arr, np.linalg.inv(covs)[np.newaxis, np.newaxis,:, : ,:], arr)
            
            exp_arg -= np.log(np.sqrt(np.linalg.det(2 * np.pi * covs)))[np.newaxis,np.newaxis,:]
            prefactor = 1
            
            #prefactor = 1 / (np.sqrt(np.linalg.det(2 * np.pi * covs)))
            
        wigner_exp_arg = np.transpose(exp_arg, [2,0,1])
        
        logwig = logsumexp(log_weights[:,np.newaxis,np.newaxis] + wigner_exp_arg, axis = 0 )
        W = prefactor*np.exp(logwig)/norm
        return W

        
    
    def get_wigner_old(self, x = None, p = None, indices = None, MP = False):
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
                    W += self.weights[i] * Gauss(np.squeeze(sigmaA), mu, x, p, MP)/self.norm
            else:
                for i, mu in enumerate(muA):
                    W += self.weights[i] * Gauss(sigmaA[i], mu, x, p, MP)/self.norm
        else: 
            if self.num_modes != 1:
                raise ValueError('State has multiple modes, please specify indices.')
            W = 0
            if len(self.covs) == 1:
                for i, mu in enumerate(self.means):
                    W += self.weights[i] * Gauss(np.squeeze(self.covs), mu, x, p, MP)/self.norm
            else:
                for i, mu in enumerate(self.means):
                    W += self.weights[i] * Gauss(self.covs[i], mu, x, p, MP)/self.norm
        
        return W

    def multimode_copy(self, n):
        """Duplicate a single mode state onto n modes.
        """
        
        # Check number of modes in state
        if self.num_modes != 1:
            raise ValueError('This is a multimode state. Can only copy make copies of single mode states.')

        if selv.num_k != self.num_weights:
            raise ValueError('Doesnt handle fast rep correctly.')
            
        means, cov, log_weights = self.means, self.covs, self.log_weights
        print('input data shape', means.shape, cov.shape, weights.shape)
        
        if self.num_covs == 1:
            new_cov = np.array([block_diag(*np.repeat(cov,n,axis = 0))])
    
        #new_weights = np.prod(np.array(list(it.product(weights.tolist(), repeat = n, ))), axis = 1)

        new_weights = np.sum(np.array(list(it.product(log_weights.tolist(), repeat = n, ))), axis = 1)
        
        new_means = np.reshape(np.array(list(it.product(means, repeat = n))), (len(log_weights)**n, n*2) )
        #new_cov = np.array([block_diag(*list(it.product(np.squeeze(cov), repeat = n)))])
    
        print('new data shape', new_means.shape, new_cov.shape, new_weights.shape)
        
        data_new = new_means, new_cov, new_weights, len(new_weights), np.sum(np.exp(new_weights))
        
        self.update_data(data_new)

    def add_state(self, state):
        """Tensor product of a state with a user-specified state in sum of Gaussian representation
        """
        #if self.num_k != self.num_weights and state.num_k != state.num_weights: 
            #raise ValueError('Doesnt handle the fast rep correctly. Need to consider the cross terms just like in the overlap functions.')
        
        means1, cov1, log_weights1 = self.means, self.covs, self.log_weights
        means2, cov2, log_weights2 = state.means, state.covs, state.log_weights
    
        k1 = self.num_k
        k2 = state.num_k
        
        #In coherent picture, covariances are the same for every weight
        if len(cov1) != 1 or len(cov2) != 1:
            new_cov = np.array([block_diag(*i) for i in list(it.product(cov1,cov2))])
        else:
            
            new_cov = np.array([block_diag(*list([np.squeeze(cov1),np.squeeze(cov2)]))])

        #Deal with different fast rep scenarios separately for correct ordering
    
        if k1 != self.num_weights and k2 != state.num_weights:  #Both are in fast rep
            
           
            nw1 = np.sum(np.array(list(it.product(log_weights1[0:k1], log_weights2[0:k2]))),axis=1) 
            nw2 = np.sum(np.array(list(it.product(log_weights1[0:k1], log_weights2[k2::]))),axis=1) 
            nw3 = np.sum(np.array(list(it.product(log_weights1[k1::], log_weights2[0:k2]))),axis=1) 
            nw4 = np.sum(np.array(list(it.product(log_weights1[k1::], log_weights2[k2::]))),axis=1) - np.log(2) #To counteract +2*np.log(2)
            nw5 = np.sum(np.array(list(it.product(log_weights1[k1::], log_weights2[k2::].conjugate()))),axis=1) - np.log(2)
    
            nm1 = np.array([np.concatenate(i) for i in list(it.product(means1[0:k1],means2[0:k2]))])
            nm2 = np.array([np.concatenate(i) for i in list(it.product(means1[0:k1],means2[k2::]))])
            nm3 = np.array([np.concatenate(i) for i in list(it.product(means1[k1::],means2[0:k2]))])
            nm4 = np.array([np.concatenate(i) for i in list(it.product(means1[k1::],means2[k2::]))])
            nm5 = np.array([np.concatenate(i) for i in list(it.product(means1[k1::],means2[k2::].conjugate()))])
    
            new_weights = np.concatenate((nw1,nw2,nw3,nw4,nw5))
            new_means = np.concatenate((nm1,nm2,nm3,nm4,nm5))
            
            num = k1*k2
    
        elif k1 != self.num_weights: #self is in fast rep

            nw1 = np.sum(np.array(list(it.product(log_weights1[0:k1], log_weights2))),axis=1) 
            nw2 = np.sum(np.array(list(it.product(log_weights1[k1::], log_weights2))),axis=1)

            nm1 = np.array([np.concatenate(i) for i in list(it.product(means1[0:k1],means2))])
            nm2 = np.array([np.concatenate(i) for i in list(it.product(means1[k1::],means2))])
            
            new_weights = np.concatenate((nw1,nw2))
            new_means = np.concatenate((nm1,nm2))
            
            num = k1

        elif k2 != state.num_weights: #state is in fast rep
            nw1 = np.sum(np.array(list(it.product(log_weights1, log_weights2[0:k2]))),axis=1) 
            nw2 = np.sum(np.array(list(it.product(log_weights1, log_weights2[k2::]))),axis=1)

            nm1 = np.array([np.concatenate(i) for i in list(it.product(means1,means2[0:k2]))])
            nm2 = np.array([np.concatenate(i) for i in list(it.product(means1,means2[k2::]))])
            
            new_weights = np.concatenate((nw1,nw2))
            new_means = np.concatenate((nm1,nm2))
            
            num = k2
            
        else: #Neither are in fast rep
            
            new_weights = np.sum(np.array(list(it.product(log_weights1, log_weights2))),axis=1) 
            #Hack to fix list of list problem
            new_means = list(it.product(means1,means2))
            new_means = np.array([np.concatenate(i) for i in new_means])
  
            num = len(new_weights)
  
           
        data_new = new_means, new_cov, new_weights, num
            
        self.update_data(data_new)
        return self


    def reduce_equal_means(self):
        r"""Merge peaks with equal means
        """
        fast = False
        if self.num_k != self.num_weights:
            fast = True
            
            
        means, cov, log_weights = self.means, self.covs, self.log_weights
        
        unique_means, idx, idx_inv  = np.unique(np.round(means,10),axis = 0, return_index = True, return_inverse=True)
        unique_means = means[idx] 
    
        unique_log_weights = np.zeros(len(idx), dtype='complex')
        
        #Compute the new weights with a high performant function doing log(sum(exp(a)))
        for i in range(len(idx)):
            lw = log_weights[idx_inv == i]
            if len(lw) != 1:
                unique_log_weights[i] = logsumexp(lw)
            else:
                unique_log_weights[i] = lw
            
        #Find number of real Gaussians, and sort them as [Re G, Im G]
        if fast: 
            
            reals = unique_means.imag == 0
            reals = reals[:,0]*reals[:,1]
            
            #Sort unique means, weights where real means go first
            means_re = unique_means[reals==True]
            weights_re = unique_log_weights[reals==True]
            means_imag = unique_means[reals==False]
            weights_imag = unique_log_weights[reals==False]
    
            unique_means = np.vstack((means_re, means_imag))
            unique_log_weights = np.hstack((weights_re, weights_imag))
            num_k = np.sum(reals == True)
            
            if out:
                print('num_k: ', num_k)
       
        if fast:
            reduced_data = unique_means, cov, unique_log_weights, num_k
        else:
            reduced_data = unique_means, cov, unique_log_weights, len(unique_log_weights)
   
        #Update the data tuple
        self.update_data(reduced_data)
        
        
    def reduce_pure(self, nmax:int, infid = 1e-6):
        """Map the state to O((nmax+1)**2) Gaussians
        Args:
            nmax : max photon number
        Returns:
            updates the state data

        """
        
        #invert the symplectic
        
        D, S = williamson(self.covs[0])
        if np.round(np.sum(np.diag(D)),1) != 2.0:
            raise ValueError('State not pure')
        
        #Remove any squeezing
        self.apply_symplectic(np.linalg.inv(S))

        data = self.means, self.covs, self.log_weights, self.num_k
        
        eps = eps_superpos_coherent(nmax, infid)
        new_data = reduce_log_pure(nmax, eps, data)

        self.update_data(new_data)
        self.get_norm()
        self.normalise()
    
        #Re-apply the squeezing
        self.apply_symplectic(S)


    def reduce_mixed(self, sd = 6, infid = 1e-6):
        """Map the state to O((nmax+1)**2) Gaussians
        Args:
            nmax : max photon number
        Returns:
            updates the state data
    
        """
        
        #invert the symplectic
        
        D, S = williamson(self.covs[0])
        nu = D - np.eye(2)
        
        #Remove any squeezing
        self.apply_symplectic(np.linalg.inv(S))
        
        #Remove thermal terms from cov
        data = self.means, self.covs - nu, self.log_weights, self.num_k
        self.update_data(data)
        
        #Find nmax from first two photon number moments
        nbar, nvar = self.get_photon_number_moments()
        nmax = int( np.ceil(nbar.real+sd*np.sqrt(nvar.real)))

        eps = eps_superpos_coherent(nmax, infid)
        
        if self.num_k != self.num_weights:
            #Perform the reduction with fast rep
            new_data = reduce_log(nmax, eps, data)
  
        else:
            #Perform the reduction with full rep
            new_data = reduce_log_full(nmax, eps, data)

        self.update_data(new_data)
        self.get_norm()
        self.normalise()
        
    
        #Re-apply the thermal noise and the squeezing
        self.covs += nu
        self.apply_symplectic(S)
        


    def sample_dyne(self, modes, shots=1, covmat = [], method = 'normal', prec= False):
        r"""Performs general-dyne measurements on a set of modes. 
        """
            
        means_quad, covs_quad, quad_ind = select_quads(self, modes, covmat)
            
        ub_ind, ub_weights, ub_weights_prob = get_upbnd_weights(means_quad, covs_quad, self.log_weights, method)

        # Perform the rejection sampling technique until the desired number of shots
        # are acquired
    
        vals = np.zeros((shots, len(modes)))
        reject_vals = []
        
        for i in range(shots):
            drawn = False
            while not drawn:
                
                # Sample an index for a peak from the upperbounding function
                # according to ub_weights_prob
                peak_ind_sample = np.random.choice(ub_ind, size=1, p=ub_weights_prob)[0]
                # Get the associated mean covariance for that peak
                mean_sample = means_quad[peak_ind_sample].real
                
                if len(covs_quad) != 1:
                    cov_sample = covs_quad[peak_ind_sample]
                else: 
                    cov_sample = covs_quad[0]
                # Sample a phase space value from the peak
                peak_sample = np.random.multivariate_normal(mean_sample, cov_sample, size =1)[0]
    
                # Calculate the probability at the sampled point
                prob_dist_val = generaldyne_probability(peak_sample, means_quad, covs_quad, self.log_weights, prec)
    
                #Calculate the upper bounding function at the sampled point
                prob_upbnd = generaldyne_probability(peak_sample, means_quad[ub_ind,:].real, covs_quad, ub_weights, prec)
                
                # Sample point between 0 and upperbound function at the phase space sample
                vertical_sample = np.random.random(size=1) * prob_upbnd
                # Keep or reject phase space sample based on whether vertical_sample falls
                # above or below the value of the probability distribution
    
                if vertical_sample > prob_dist_val:
                    reject_vals.append(peak_sample)
                if vertical_sample <= prob_dist_val:
                    drawn = True
                    vals[i] = peak_sample
        
        return vals, np.array(reject_vals)
            
    
    def sample_dyne_gaussian(self, modes, shots = 1, covmat = [], factor = 0, prec =False):
        r"""Performs general-dyne measurements on a set of modes using a Gaussian 
        upper bounding function based on the first and second moments of the state. 
        
        """
        means_quad, covs_quad, quad_ind = select_quads(self, modes, covmat)
        cov_ub, mean_ub, scale = get_upbnd_gaussian(self, means_quad, covs_quad, quad_ind, prec)
            
        #Perform rejection sampling with the single guassian upper bounding function
        vals = np.zeros((shots, len(modes)))
        reject_vals = []        
        
        for i in range(shots):
            drawn = False
            while not drawn:
              
                #Draw a sample from the Gaussian
                sample = np.random.multivariate_normal(mean_ub, cov_ub, size =1)[0]
                if factor ==0:
                    prefactor = 1/np.sqrt(2*np.pi*np.linalg.det(cov_ub))
                    prob_upbnd = generaldyne_probability(sample, mean_ub, cov_ub, np.array([np.log(scale/prefactor)])) 
                else:
                    prob_upbnd = generaldyne_probability(sample, mean_ub, cov_ub, np.array([np.log(factor)]))
                    
                
                y = np.random.random(size=1)*prob_upbnd 
    
                prob_dist_val = generaldyne_probability(sample, means_quad, covs_quad, self.log_weights, prec)
             
                if y > prob_dist_val:
                    reject_vals.append(sample)
                    
                elif y <= prob_dist_val:
                    drawn =True
                    vals[i] = sample
    
        return vals, np.array(reject_vals)
    
    
    