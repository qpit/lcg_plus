# Copyright © 2025 Technical University of Denmark

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from lcg_plus.interferometers.symplectics import rotation
from lcg_plus.interferometers.ops import apply_symplectic
from lcg_plus.states.from_sf import prepare_cat_bosonic, prepare_gkp_bosonic
from lcg_plus.states.coherent import outer_coherent, eps_fock_coherent, eps_superpos_coherent
from lcg_plus.states.wigner import get_wigner_coherent
from strawberryfields.backends.states import BaseBosonicState
from lcg_plus.plotting import plot_wigner_marginals
import matplotlib.pyplot as plt
from scipy.special import factorial
import strawberryfields as sf
from strawberryfields.backends.bosonicbackend import ops

from lcg_plus.interferometers.parameters import gen_interferometer_params
from lcg_plus.interferometers.construct import build_interferometer
from lcg_plus.measurements.photon_counting import post_select_fock_coherent

from thewalrus.symplectic import xpxp_to_xxpp, xxpp_to_xpxp

class PureBosonicState:
    """Store a pure bosonic state as superposition of coherent states by tracking its amplitudes and coherent state amplitudes
    """
    def __init__(self, pure_data : tuple):
        """
        pure_data = [means, covs, coeffs]
        """
        self.pure_data = pure_data
        self.coeffs = pure_data[2]
        self.covs = pure_data[1]
        self.means = pure_data[0]
        self.alphas = pure_data[0]/np.sqrt(sf.hbar*2)
        self.num_alphas = len(self.means)

    def get_dm_form(self):
        means = []
        covs = np.eye(2)
        cs = self.coeffs
        weights = []

        for i in range(self.num_alphas):
            for j in range(self.num_alphas):
                
                mij, cov, cij = outer_coherent(self.alphas[i], self.alphas[j])
                weights.append(cs[i]*np.conjugate(cs[j])*cij)
                #weights[i,j] *= cij
                means.append(mij)
        
        self.data = np.array(means), covs, np.array(weights)/np.sum(weights)
        
    def get_fock_probs(self, cutoff = 10):
        ns = range(cutoff)
        ps = np.zeros(cutoff,dtype='complex')
        cs = self.coeffs
        a = self.alphas
        for n in ns: 
            
            for i in range(self.num_alphas):
                ps[n] += cs[i]* np.exp(-np.abs(a[i])**2/2) * a[i]**n/ np.sqrt(factorial(n))
                
        
        self.fock_probs = np.abs(ps)**2

def coherent_overlap(mu1, mu2):
    """DOUBLE CHECK THIS
    mu1 = alpha + 1j beta
    mu2 = delta +1j gamma
    """
    delta_m = mu1 - mu2 
    delta_p = mu1 + mu2
    exparg = - delta_m.imag**2 - delta_m.real**2 + 2*1j*delta_m.imag*delta_p.real
    
    return np.exp(1/4*exparg)

def fock_coherent_pure(N, inf=1e-4, eps = None):
    """ OBS: PROBLEMS WITH THE NORMALISATION
    
    Generate the wavefunction data for a Fock state N in the coherent state representation.
    
    Args:
        N (int): fock number
        infid (float): infidelity of approximation
        eps (None or float): coherent state amplitude, takes presidence over infid if specified.
    Returns:
        [mus, covs, coeffs] : tuple, data tuple for the pure state

            In particular, mus = sqrt(2*hbar)*alphas, so real part is disp in x, imag part is disp in p 
        
    See Eq 28 of http://arxiv.org/abs/2305.17099 for expansion into coherent states.
    """
    
    alphas = []
    theta = 2*np.pi/(N+1)

    coeffs = []
    if N == 0:
        alphas.append(0)
        coeffs.append(1)
        
    else:
        
        if not eps:
            eps = eps_fock_coherent(N, inf)
        
        for k in np.arange(N+1):
    
            alpha = eps * np.exp(1j * theta * k) 
            alphas.append(alpha)
            coeffs.append(np.exp(-theta*1j*N*k))

        coeffs = np.array(coeffs)
        factor = np.sqrt(factorial(N))/(N+1) * np.exp(eps**2/2)/eps**N
        norm_est = 1+eps**(2*(N+1))/factorial(N+1)
        #print(norm_est)
        coeffs *= factor
        #coeffs /= np.sqrt(norm_est)
        #coeffs /= np.sqrt(np.sum(np.abs(coeffs)**2))

    #Normalise the coefficients
    
    norm = 0
    for k, ck in enumerate(coeffs):
        norm += np.abs(ck)**2
        l = k+1
        if len(coeffs) - 1 >= l > k: 
            cl = np.conjugate(coeffs[l])
            norm += 2*np.real(ck*cl * coherent_overlap(alphas[k], np.conjugate(alphas[l])))
            l+=1
    #print(norm)
    coeffs/=np.sqrt(norm)

    covs = sf.hbar/2*np.eye(2) #Vacuum state
    
    return np.sqrt(2*sf.hbar)*np.array(alphas), covs, coeffs


def gkp_coherent_pure(n, type, infid=1e-4):
    """NORMALISATION
    """
    rho = gkp_nonlinear_squeezing_operator(n, 1, type=type)

    w, v = np.linalg.eigh(rho)
    
    coeffs_n = v[:,0] #eigs always sorted from lowest to highest eigenvalue, choose lowest

    alphas, coeffs = fock_superpos_coherent(coeffs_n, infid)


    return np.sqrt(sf.hbar*2)*alphas, sf.hbar/2*np.eye(2), weights
    
    
def prepare_coherent(alpha):
    """DOUBLE CHECK THIS
    """
    weights = np.array([1])
    means = np.array([np.array([alpha.real, alpha.imag])])*np.sqrt(sf.hbar*2)
    covs = np.array([np.eye(2)])
    return BaseBosonicState([means, covs, weights], 1, 1)


def customsqrt(alpha):
    """Returns the positive sqrt of a complex number

    Args: 
        alpha : complex, r*exp(theta)
    Returns:
        (np.sqrt(r), theta/2) : tuple, 
    """
    r = np.abs(alpha)
    theta = np.angle(alpha)
    #r, theta = polar(alpha)
    if theta < 0: 
        theta += 2*np.pi
    return (np.sqrt(r), theta/2)

#Get the phase matrix from the covariance matrix

def decomp_cov(cov : np.ndarray):
    """Decompose covariance matrix into cov = [[A, C],[C.T, B]]

    Args: 
        cov : ndarray, covariance matrix in xxpp
    Returns:
        A, B, C : tuple
    """
    #Get number of modes
    N = int(cov.shape[-1])
    Nhalf = int(N/2)
    A = cov[:, 0:Nhalf,0:Nhalf]
    B = cov[:, Nhalf::, Nhalf::]
    C = cov[:, 0:Nhalf,Nhalf::] #xp correlations
    return A, B, C

def get_phase_mat(cov : np.ndarray):
    """Get phase matrix from covariance matrix

    Args:
        cov : ndarray, covariance matrix in xxpp

    Returns: 
        Phi : ndarray, phase matrix
    """
    A, B ,C = decomp_cov(cov)
    Ainv = np.linalg.inv(A)
    return Ainv+1j*np.transpose(C,axes=[0,2,1]) @ Ainv

def chop_phase_mat(Phi, idx):
    """Chop phase matrix into A and B modes

    Args:
        Phi (ndarray): phase mat
        idx (int): index of mode B
    Returns:
    (phiA, phiBA, phiB): partitioned phase matrix
    """
    phiA = np.delete(Phi, idx, axis = 1)
    phiA = np.delete(phiA, idx, axis = 2)
    phiB = Phi[:,idx,idx]
    phiBA = np.delete(Phi[:,idx,:],idx,axis=1)
    return (phiA, phiBA, phiB)

def chop_disp_vec(mu, idx):
    """
    Args:
        mu (ndarray): complex array containing x (real) and p (imag) displacements
        idx (int): index of mode B
    Returns:
        (muA, muB) : partitioned displacements
    """
    muA = np.delete(mu,idx,axis=1)
    muB = mu[:,idx]
    return (muA, muB)

def post_select_phase_mat(Phi, mu, phiM, muM, idx):
    """DEAL WITH NORMS

    Post select the phase matrix and disp vector with PVM with phasemat phiM and disp vector muM
    See overleaf bosonicplus wavefunction section for formula and derivation
    Returns
        phi_A_tilde : np.ndarray, post-selected phase mat of remaining modes 
        s_prime : np.ndarray, post-selected p displacements
        t_prime : np.ndarray, post-selected x displacements
        norm : , overall norm constant
        exparg : np.ndarray, exparg in remainder of norm constant
        phase : phase, from complex part of exparg
    """
    nmodes = int(np.shape(Phi)[1]/2)
    #Chop up Phi
    phiA, phiBA, phiB = chop_phase_mat(Phi,idx)
    muA, muB = chop_disp_vec(mu,idx)
 
    phiB = np.squeeze(phiB) #We always measure one mode at a time, so this also has shape (1,1,1)
    phiMc = np.conjugate(phiM)
    
    phiW = phiB+phiMc
    #print('phiW', phiW, 2*np.cosh(r)**2) #For two mode squeezing example
    
    w = 1/phiW*(phiB*muB.real[np.newaxis,:] + phiMc*muM.real[:,np.newaxis])
    #print('w', w, 1/(np.cosh(r)**2)*muM.real/2, w.shape)
   
    exparg1 = -1/2*((muB.real[np.newaxis,:])**2*phiB + (muM.real[:,np.newaxis])**2*phiMc - w**2*phiW)
    #print('exparg1',exparg1, (muM.real/2)**2 * (1/(np.cosh(r)**2)-2))
    
    
    
    delta_imag = muB.imag[np.newaxis,:] - muM.imag[:,np.newaxis]
    #print('delta_imag**2', delta_imag**2, [i**2 for i in delta_imag])
    exparg2 = 1j*delta_imag*w - delta_imag**2/(2*phiW)
    #print('exparg2', exparg2, (muM.imag/2)**2 * 1/(np.cosh(r))**2 - 1j*2*(muM.real/2)*(muM.imag/2)*1/(np.cosh(r)**2))
    exparg3 = 1j/phiW* delta_imag * np.einsum("...j,...j", phiBA, muA.real)
    

    phi_A_tilde = phiA - (phiBA.T @ phiBA)/phiW 
   
    #print('muB - muM:',delta_imag.shape, phiBA.shape)
    
    s_prime = muA.imag - 1/phiW* delta_imag @ phiBA
    
    delta_re = muB.real[np.newaxis,:] - w
    t_prime = muA.real + np.einsum("...jk,...k", np.linalg.inv(phi_A_tilde) ,delta_re @ phiBA)

    exparg4 = 1/2*delta_re**2 * np.squeeze((phiBA @ np.linalg.inv(phi_A_tilde) @ phiBA.T))

    exparg = exparg1+exparg2+exparg3+exparg4

    phase = exparg.imag    

    delta = muM[np.newaxis,:] - muB[:, np.newaxis]
    

    #print('phase', phase , -2*1j*muM.real/2*muM.imag/2*1/(np.cosh(r))**2)

    #Constants
    N1 = (np.linalg.det(Phi.real)/(np.pi)**nmodes)**(1/4)
    #print('N1', N1.shape)
    N2 = (phiM.real/(np.pi))**(1/4)
    #print('N2',N2.shape)
    
    #Getting the positive sqrt of a complex number correctly
    a, phi = customsqrt(phiW) #Will always just be a number
    

    N3 = np.sqrt(2*np.pi)/(a * np.exp(1j*phi))
    #print(N3)
    N3 = np.sqrt(2*np.pi/phiW)
    #print(N3)
    N3 = np.sqrt(1/phiW)
    norm = N1*N2*N3
    norm = N3
    #norm = 1
    
    #print('N3', N3.shape)
    #N4 = np.exp(-1/2*((muB.real[np.newaxis,:])**2*phiB + (muM.real[:,np.newaxis])**2*np.conjugate(phiM) - w**2*phiW))
    #print('N4', N4.shape)
    
    #If we want to dealing with case where displacements are complex
    comp = False
    if comp: 
        print('Doing complex part:', comp)
    
        alpha = s_prime.real
        beta = s_prime.imag
        delta = t_prime.real
        gamma = t_prime.imag
        
        t_prime = delta - np.einsum("...jk,...k",np.linalg.inv(phi_A_tilde), beta)
        s_prime = alpha + np.einsum("...jk,...k",phi_A_tilde, gamma)
        #print(t_prime.shape)
        
        #print(alpha.shape, beta.shape)
        
        arg1 = 1/2*np.einsum("...j,...jk,...k", gamma, phi_A_tilde, gamma)
        arg2 = -1j*np.einsum("...j,...jk,...k", gamma, phi_A_tilde, delta)
        arg3 = -1/2*np.einsum("...j,...jk,...k",delta, phi_A_tilde, delta)
        dlta = np.einsum("...jk,...k", phi_A_tilde, delta) - beta
        arg4 = 1/2*np.einsum("...j,...jk,...k",dlta,np.linalg.inv(phi_A_tilde), dlta)
        
        #print(arg1.shape, arg2.shape, arg3.shape, arg4.shape)
        exparg_c = np.array([arg1+arg2+arg3+arg4]).T
        print(exparg_c.shape, exparg.shape)
        #print(exparg_c.shape)

        exparg += exparg_c
        
        #M4 = np.array([np.exp(exparg_c)]).T
    #print(exparg1.shape)
    

    

    # Now, get the probability
    
    
    #print('M4', M4.shape)
    
    #norm = 1
    #norm = N1*N2*N4
    
    #normfull = N1*N2*N3*N4*M1*M2*M3*M4
    
    #phase = np.angle(norm)
    #phasefull = np.angle(normfull)
    #phase = np.angle(normfull)
    #print('Phase from imag constants equal to phase from all constants: ', np.allclose(phase, phasefull))

    
    return phi_A_tilde, t_prime, s_prime, norm, exparg, phase
    

def post_select_fock_pure(data, n, mode, inf = 1e-4):
    """TO DO: INCLUDE PROBABILITY
    Args: 
        data : tuple, [means, covs, weights] (pure)
        n : int, fock number
        mode : int, mode that is measured
        inf: float, infidelity of fock approx
    Returns:
        tuple
    """
    means, covs, weights = data
    #Obs: what's the convention for the means? Is it size (nmodes) or (2*nmodes), i.e. is the real or complex elements? 
    sigma_xxpp = np.array([xpxp_to_xxpp(i) for i in covs])

    mu = means[:,0::2]+1j*means[:,1::2] #OBS: these are (x, nmodes) shape. The real part is the disp in x and the imaginary part is the disp in p
    #But, is not the alpha, as there is no division by sqrt(2*hbar)

    Phi = get_phase_mat(sigma_xxpp)

    #The fock state PVM in coherent state representation
    means_f, sigma_f, coeffs_f = fock_coherent_pure(n, inf)

    print(np.shape(means_f), np.shape(sigma_f), np.shape(coeffs_f))

    #Get phase matrix of PVM
    phiM = get_phase_mat(np.array([sigma_f])) #Will always be (1,1,1), so can squeeze it
    phiM = np.squeeze(phiM)

    phi_A_tilde, t_prime, s_prime, norm, exparg,phase = post_select_phase_mat(Phi, mu, phiM, means_f, mode)

    return coeffs_f, phi_A_tilde, t_prime, s_prime, norm, exparg, phase

    
def get_prob(coeff, phi, t, s, norm, exparg,phase):
    """NOT WORKING PROPERLY, LIKELY DUE TO COEFFS, AND NORMS

    Args:
        coeffs : np.ndarray, pure state coefficients of coherent state superposition
        phi : np.ndaarray, list of one phase mat
        s: np.ndarray, p disps
        t: np.ndarray, x disps 
        norm: overall norm factor
        exparg : 
        phase: phase from complex part of exparg
    Returns: 
        p (float): probability
    """

    #print(phi_A_tilde.shape)
    #print(t_prime.shape)
    #print(s_prime.shape)
    #print(exparg.shape)
    #print(exparg.shape)
    
    nmodes = int(phi.shape[2])
    #print(nmodes)
    #expargk = -1/2*np.einsum("...j,...jk,...k",t_prime, phi_A_tilde, t_prime)
    #expargl = -1/2*np.einsum("...j,...jk,...k",np.conjugate(t_prime), np.conjugate(phi_A_tilde), np.conjugate(t_prime))

    phi_c = np.conjugate(phi)
    phi_rinv = np.linalg.inv(phi.real)
    tc = np.conjugate(t)
    sc = np.conjugate(s)
    prob = 0

    #Start with diag term
    
    #ck = np.abs(coeff)[np.newaxis,:]
    #ark = np.abs(np.exp(exparg))

    
    #sk = s - sc
    
    #tk = 1/2*(t @ phi + tc @ phi_c) @ phi_rinv
    #skk = s_prime - np.conjugate(s_prime)
    #sk = 2*1j*s.imag


    #print(norm.shape)
    #print(tkk1,tkk)
    #K_exparg1 = -1/2 * t_prime **2 @ phi_A_tilde - 1/2*np.conjugate(t_prime)**2 @ np.conjugate(phi_A_tilde)
    
    #K_exparg = - (t**2 @ phi ).real + 0j


    #K_exparg += 1/2*tk**2 @ phi.real

    #C_exparg = 1j*sk*tk - 1/2* sk**2 @ phi_rinv
    

    #expargskk = arg1 + K_exparg + C_exparg
    #Dkk =  ck @ ark * np.abs(np.exp(K_exparg+C_exparg))
    #print(Dkk.shape)
    #prob += np.sum(Dkk)
    #imax = len(ck)-1
    #print(t)
    #print(s)
    for i, ti in enumerate(t):
        #j = i+1
        si = s[i]
        ci = coeff[i]
        for j, tj in enumerate(tc):
            
        #while j <= imax: 
            #if j > i:
            sj = sc[j]
            tij = 0.5*(ti * phi + tj * phi_c ) * phi_rinv
            #print(tij.shape)
            sij = si - sj
            cj = np.conjugate(coeff[j])
            
            K_expargij = -1/2 * ti **2 * phi - 1/2*tj**2 * phi_c 
            
            K_expargij += 1/2*tij**2 * phi.real
            #K_expargij += tij**2 * phi.real
    
            #C_expargij = 1j*sij * tij -0.5*sij**2 * phi_rinv
            C_expargij = 1j*sij * tij -0.5*sij**2 * phi_rinv
           
            arij = exparg[i] + np.conjugate(exparg[j])
           
            
            #prob += 2*(ci *cj*np.exp(arij + K_expargij + C_expargij)).real 
            prob += ci*cj*np.exp(arij + K_expargij + C_expargij)
            #print('prob:', np.exp(arij + K_expargij + C_expargij))
            #j+=1 
    #print(prob.shape)
    #prob *= norm * np.conjugate(norm)* np.sqrt(np.pi**nmodes/(np.linalg.det(phi.real)))

    prob *= norm * np.conjugate(norm)
    #prob *= norm **2 * np.sqrt(np.pi**nmodes/(np.linalg.det(phi.real)))

    
    return prob 
           
        
        
    
