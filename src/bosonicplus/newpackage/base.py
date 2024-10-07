import numpy as np
from thewalrus.symplectic import xpxp_to_xxpp, xxpp_to_xpxp

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
        self.data = [self.means, self.covs, self.weights]
        self.num_weights = len(self.weights)
        self.num_covs = len(self.covs) #Relevant for faster calculations
        self.ordering = 'xxpp'
        

    def update_data(self, new_data : tuple):
        """Insert a custom data tuple
        To do: what if new_data is in a different ordering?
        """
        self.data = new_data
        self.means = self.data[0] 
        self.covs = self.data[1]
        self.weights = self.data[2]
        self.num_weights = len(self.weights)
        self.num_covs = len(self.covs)
        self.num_modes = int(np.shape(self.means)[-1])



    def to_xpxp(self):
        """Change the ordering from xpxp to xxpp
        """
        means, covs, weights = self.data
        means = [xxpp_to_xpxp(i) for i in means]
        covs = [xxpp_to_xpxp(i) for i in covs] #quick workaround
        self.update_data([means, covs, weights])
        self.ordering = 'xpxp'

    def to_xxpp(self):
        """Change the ordering from xpxp to xxpp
        """
        means, covs, weights = self.data
        means = [xpxp_to_xxpp(i) for i in means]
        covs = [xpxp_to_xxpp(i) for i in covs] #quick workaround
        self.update_data([means, covs, weights])
        self.ordering = 'xxpp'
    
    def apply_symplectic(self, S : np.ndarray):
        """ Apply symplectic to data
            S (array) : symplectic matrix

            To do: check covs update when num_covs != 1

            Note that symplectics from thewalrus are given in xxpp ordering
        """
        means, covs, weights = self.data
        
        #First check that S has the correct dimensions
        if np.shape(S)[0] != int(np.shape(means)[-1]):
            raise ValueError('S must must be 2nmodes x 2nmodes. ')
        if self.num_covs == 1: 
            new_covs = S @ covs @ S.T
        else:
            #Didn't test if this works
            new_covs = np.einsum("...jk,...kl,...lm", S[np.newaxis,:], covs, (S.T)[np.newaxis, :])

        new_means = np.einsum("...jk,...k", S[np.newaxis,:], means)
    
        #Update data
        self.update_data([new_means, new_covs, weights])
        
        
    def apply_displacement(self, d: np.ndarray):
        r""" d must be in the same ordering (xxpp) or (xpxp)
        To do: add test of d shape compatibility
        """
        means, covs, weights = self.data
    
        #Update data
        self.update_data([means+d, covs, weights])
    

    def apply_loss(self, etas, nbars):
        """Apply loss to (multimode) state in data

        To do: add etas and nbars shape check, add case where num covs !=1
    
        Gaussian state undergo a loss/thermal loss channel in the following way:
            cov = X @ cov @ X.T + Y
            means = X @ means
    
        Args:
            etas (array): array giving transmittivity of each mode in data
            nbars (array): array giving number of photons in environment each mode is coupled to
    
        Returns:
           updates self.data
        """
        num_modes = self.num_modes
        means, cov, weights = self.data

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
        self.update_data([means, cov, weights])