from scipy.optimize import basinhopping, minimize
from bosonicplus.operations.circuit_parameters import gen_interferometer_params, params_to_1D_array, unpack_params, params_to_dict
import numpy as np
from bosonicplus.conversions import dB_to_r, r_to_dB
from bosonicplus.cost_functions import symm_effective_squeezing
import pickle
import os.path

class GBS_optimizer:

    def __init__(self, num_modes = 2, 
                 pattern = [1],
                 bs_arrange = 'inv_cascade', 
                 setting = 'no_phase',
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

        #set the optimizer bounds
        if self.setting == 'no_phase':
            self.bounds = [(-self.r_max,self.r_max)]*self.num_modes + [(0.1, np.pi/2-0.1)] * self.num_bs
        elif self.setting == 'two_mode_squeezing':
            self.bounds = list(chain.from_iterable(zip([(0,self.r_max)]*self.num_bs, [(-np.pi, np.pi)] * self.num_bs)))

        #Calculate the initial cost function value
        self.init_costf = self.costf(self.guess, *self.costf_args)


    def run_global_optimisation(self,  
                         method = 'L-BFGS-B', 
                         maxiter = 500, 
                         niter = 50, 
                         stepsize = 2,
                         tol = 1e-4,
                         disp = True):
        """
        method : from basinhopping, SLSQP or L-BFGS-B works best 
        niter : see basinhopping
        setpsize : see basinhopping
        disp (bool) print status of optimisation

        """

        minimizer_kwargs = {'args': self.costf_args, 
                            'method': method,
                            'jac' : self.gradients,
                            'tol': tol,
                            'bounds': self.bounds,
                             'options': {'maxiter': maxiter}
                           }
                
        
        params_init = self.guess
        
        #res = shgo(infidelity_nmode_GBS, bounds = bounds, args = (nmodes, n, target, T))
        
        
        res = basinhopping(self.costf, x0 = params_init, niter = niter,
        minimizer_kwargs = minimizer_kwargs, disp = disp, stepsize=stepsize) 

        self.result = res
        self.res_dict = params_to_dict(res.x, self.num_modes, self.bs_arrange, self.setting)


    def run_local_optimisation(self,  
                         method = 'L-BFGS-B',  
                         maxiter = 500,
                         disp = True):
        """
        method : , SLSQP or L-BFGS-B works best 
        bounds : set your own custom bounds. 

        """
        
        params_init = self.guess
        
        #res = shgo(infidelity_nmode_GBS, bounds = bounds, args = (nmodes, n, target, T))
        
        res = minimize(self.costf, 
                       x0=params_init, 
                       args=self.costf_args, 
                       method=method, 
                       jac=self.gradients, 
                       bounds=self.bounds,
                       options = {'maxiter': maxiter, 'disp':disp})
        
        self.result = res
        self.res_dict = params_to_dict(res.x, self.num_modes, self.bs_arrange, self.setting)


class GBS_opt_light:
    """A lighter version of GBS_optimizer. Intended to store many optimisation results with the same circuit settings.
    """
    def __init__(self, nmodes, bs_arrange, setting, pattern):
        #Metadata
        self.bs_arrange = bs_arrange
        self.nmodes = nmodes
        self.setting = setting
        self.pattern = pattern
        
        self.params = []
        self.costfs = []
        self.num_opts = 0
    
    def add_opt(self, opt):
        self.params.append(opt.result.x)
        self.costfs.append(opt.result.fun)
        self.num_opts += 1

def run_opts(nmodes, num_opts, cutoff, niter, bs, costfs, patterns, inf, costf_lattice, setting, pPNR, nbars, etas):
    
    for i, bs_arrange in enumerate(bs):
        for j, costf in enumerate(costfs): 
            gradients = j == 1 #gradients for second costf
            fast = j == 0 #fast rep for first costf
            
            np.random.seed(28) #Each costf uses the same initial guesses
    
            for k, pattern in enumerate(patterns): 
                print(bs_arrange, costf, pattern)
                fname = f'{bs_arrange}_{gradients}_{pattern}_opt.pickle'
                if os.path.isfile(fname):
                    print('file already exits.')
                    
                else:
                    opt_light = GBS_opt_light(nmodes, bs_arrange, setting, pattern)
        
                    if np.sum(pattern) != 0:
                        num = 0
                        while num < num_opts: 
                    
                            opt = GBS_optimizer(nmodes,
                                                list(pattern),
                                                bs_arrange,
                                                setting,
                                                costf,
                                                costf_lattice,
                                                pPNR,
                                                gradients,
                                                inf,
                                                etas,
                                                nbars,
                                                fast  
                                               )
                            opt.set_initial_guess()
                           
                            
                            opt.run_global_optimisation(disp = False, niter = niter)
                            print(f'global optimum {num}', opt.result.fun)
                            opt_light.add_opt(opt)
    
                            num += 1
                        with open(f'{bs_arrange}_{gradients}_{pattern}_opt.pickle', 'wb') as handle:
                            pickle.dump(opt_light, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                    
    
        

  