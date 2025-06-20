from scipy.optimize import basinhopping, minimize
from bosonicplus.operations.circuit_parameters import gen_interferometer_params, params_to_1D_array, unpack_params, params_to_dict
import numpy as np
from bosonicplus.conversions import dB_to_r, r_to_dB
from bosonicplus.cost_functions import symm_effective_squeezing
class GBS_optimizer:

    def __init__(self, num_modes, 
                 bs_arrange = 'inv_cascade', 
                 setting = 'no_phase',
                 pattern = [1],
                 costf = symm_effective_squeezing,
                 costf_lattice = 's',
                 pPNR = False,
                 gradients = False,
                 inf = 1e-4,
                 etas = None,
                 nbars = None,
                fast =False):
        
        """Setup the optmizer parameters
        num_modes (int)
        bs_arrange (str)
        pPNR (False or int)
        inf (float) : infidelity of PNRD approx
        """
        if isinstance(num_modes ,int):
            self.num_modes = num_modes
        else: raise ValueError('num_modes must be int.')
        
        bs_string = ['Clements', 'inv_cascade', 'cascade']
        
        if bs_arrange in bs_string:
            self.bs_arrange = bs_arrange
        else: 
            raise ValueError('beam splitter arrangement can be either Clements, inv_cascade or cascade.')

        setting_string = ['no_phase', 'two_mode_squeezing']
        if setting in setting_string:
            self.setting = setting
        else:
            raise ValueError('setting can either be no_phase or two_mode_squeezing.')
        
        #Photon number pattern
        if isinstance(pattern, np.ndarray) or isinstance(pattern, list):
            
            if len(pattern) != num_modes - 1: 
                raise ValueError('length of pattern must be nmodes - 1 .')
            
            self.pattern = pattern
        else: raise ValueError('pattern must be either np.ndarray or list.')

        if type(inf) == float and inf < 1: 
            self.inf = inf
        else: raise ValueError('inf must be real and less than 1.')

        #Pseudo-PNRD functionality
        if pPNR == False:
            self.pPNR = pPNR
        elif pPNR:
            if isinstance(pPNR, int):
                self.pPNR = pPNR #Number of 
            else: raise ValueError('pseudo PNR setting must either be None (not in use) or an integer.')

        self.gradients = gradients
        self.costf = costf

        if isinstance(etas, type(None)): 
            self.etas = etas
        
        elif isinstance(etas,np.ndarray):
            if len(etas) != num_modes:
                raise ValueError('etas array must be same length as number of modes.')
        
            self.etas = etas
            
        else: raise ValueError('loss must either be None, np.ndarray')

        
        if isinstance(nbars, type(None)):
            self.nbars = nbars
            
        elif isinstance(nbars,np.ndarray):
            
            if len(nbars) != num_modes:
                raise ValueError('nbar array must be same length as number of modes.')
            
            self.nbars = nbars

        else: raise ValueError('nbars must either be None, np.ndarray')

        self.fast = fast

        #setup cost args and kwargs
        self.costf_args = (self.num_modes, 
                             self.pattern,
                             self.bs_arrange,
                             self.setting,
                             self.etas,
                             self.nbars,
                             self.pPNR, 
                             self.gradients, 
                             self.inf, self.fast, costf_lattice)
        
                
        #self.costf_kwargs = {'lattice': costf_lattice}


    def set_initial_guess(self, r_max_dB = -15, phases = False, disp = False, params = None):
        """Set the initial guess for the optimisation.

            params : dict (define your own guess) 
        """
        if phases:
            raise ValueError('Extra phases not yet compatible with optimisation.')
        if disp:
            raise ValueError('Displacements not yet compatible with optimisation.')
            
        if isinstance(params, type(None)): 
            #if setting == 'no_phase':
            params = gen_interferometer_params(self.num_modes, r_max_dB, self.bs_arrange, self.setting)
           # elif setting == 'takase':
                
            
        self.guess_dict = params
        self.guess = params_to_1D_array(self.guess_dict, self.setting)
        self.num_params = len(self.guess)
        
        self.r_max = np.abs(dB_to_r(r_max_dB))
        self.num_bs = len(self.guess_dict['bs'])

        #Calculate the initial cost function value
        self.init_costf = self.costf(self.guess,*self.costf_args)


    def run_global_optimisation(self,  
                         method = 'L-BFGS-B', 
                         maxiter = 500, 
                         niter = 50, 
                         bounds = None,
                         out = True):
        """
        method : from basinhopping, SLSQP or L-BFGS-B works best 
        niter : see basinhopping
        setpsize : see basinhopping
        bounds : set your own custom bounds. 
        out (bool) print status of optimisation

        """

        if isinstance(bounds, type(None)):
            if self.setting == 'no_phase':
                #bounds = [(-self.r_max,self.r_max)]*self.num_modes + [(0, np.pi/2)] * self.num_bs + [(-np.pi, np.pi)]*self.num_bs
                bounds = [(-self.r_max,self.r_max)]*self.num_modes + [(0.1, np.pi/2-0.1)] * self.num_bs
            elif self.setting == 'two_mode_squeezing':
                bounds = list(chain.from_iterable(zip([(0,self.r_max)]*self.num_bs, [(-np.pi, np.pi)] * self.num_bs)))
                #bounds = [(0,self.r_max)]*self.num_bs + [(-np.pi, np.pi)] * self.num_bs
        
        minimizer_kwargs = {'args': self.costf_args, 
                            'method': method,
                            'jac' : self.gradients,
                            'tol': 1e-4,
                            'bounds': bounds,
                             'options': {'maxiter': maxiter}
                           }
                
        
        params_init = self.guess
        
        #res = shgo(infidelity_nmode_GBS, bounds = bounds, args = (nmodes, n, target, T))
        
        
        res = basinhopping(self.costf, x0 = params_init, niter = niter,
        minimizer_kwargs = minimizer_kwargs, disp =True, stepsize=2) 

        
            
        #self.result = res
        self.result = res
        self.res_dict = params_to_dict(res.x, self.num_modes, self.bs_arrange, self.setting)


    def run_local_optimisation(self,  
                         method = 'L-BFGS-B',  
                         maxiter = 500,
                         bounds = None):
        """
        method : , SLSQP or L-BFGS-B works best 
        bounds : set your own custom bounds. 

        """

        if isinstance(bounds, type(None)):
            if self.setting == 'no_phase':
            #bounds = [(-self.r_max,self.r_max)]*self.num_modes + [(0, np.pi/2)] * self.num_bs + [(-np.pi, np.pi)]*self.num_bs
                bounds = [(-self.r_max,self.r_max)]*self.num_modes + [(0.1, np.pi/2-0.1)] * self.num_bs
            elif self.setting == 'two_mode_squeezing':
                bounds = list(chain.from_iterable(zip([(0,self.r_max)]*self.num_bs, [(-np.pi, np.pi)] * self.num_bs)))
                #bounds = [(0,self.r_max)]*self.num_bs + [(-np.pi, np.pi)] * self.num_bs

        
        
        params_init = self.guess
        
        #res = shgo(infidelity_nmode_GBS, bounds = bounds, args = (nmodes, n, target, T))
        
        res = minimize(self.costf, x0=params_init, args=self.costf_args, method=method, jac=self.gradients, bounds=bounds,options = {'maxiter': maxiter, 'disp':True})
        
        self.result = res
        self.res_dict = params_to_dict(res.x, self.num_modes, self.bs_arrange, self.setting)

  