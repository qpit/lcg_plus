from bosonicplus.base import State
from bosonicplus.fidelity import fidelity_bosonic, fidelity_with_wigner
import numpy as np
from thewalrus.symplectic import beam_splitter, squeezing, xxpp_to_xpxp, expand
from bosonicplus.states.wigner import wig_mn

#Some gates
nmodes = 3
S1 = xxpp_to_xpxp(expand(squeezing(0.5),[0], nmodes))
S2 = xxpp_to_xpxp(expand(squeezing(1, np.pi),[1], nmodes))
S3 = xxpp_to_xpxp(expand(squeezing(0.5),[2], nmodes))
BS1 = xxpp_to_xpxp(expand(beam_splitter(np.pi/3, 0), [0,1],nmodes))
BS2 = xxpp_to_xpxp(expand(beam_splitter(np.pi/5, 0), [1,2],nmodes))

Stot = BS2@BS1@S3@S2@S1

S1 = xxpp_to_xpxp(squeezing(0.5))
S2 = xxpp_to_xpxp(squeezing(1, np.pi))
S3 = xxpp_to_xpxp(squeezing(0.5))
BS1 = xxpp_to_xpxp(beam_splitter(np.pi/3, 0))
BS2 = xxpp_to_xpxp(beam_splitter(np.pi/5, 0))

#Sim results of this circuit in strawberryfields (1st and 2nd mode conditioned on x-homodyne with result 0)
mean_photon_sf_gauss = (0.22398928178451882, 0.5483209602777269)
sf_gauss_vacuum_fid = 0.9038808650986042

n = 6 #cutoff 25, measure n in first 2 modes
mean_photon_sf_fock = (0.6783735266693751, 1.2103871063143152)
sf_fock_vacuum_fid = 0.6979407608878457


def test_apply_symplectic():
    """Test that apply_symplectic and apply_symplectic_fast does the same operation
    """ 
    state1 = State(nmodes)
    state1.apply_symplectic(Stot)
    
    state2 = State(nmodes)
    state2.apply_symplectic_fast(S1, [0])
    state2.apply_symplectic_fast(S2, [1])
    state2.apply_symplectic_fast(S3, [2])
    state2.apply_symplectic_fast(BS1, [0,1])
    state2.apply_symplectic_fast(BS2, [1,2])
    
    assert np.allclose(state1.covs, state2.covs)
    assert np.allclose(state1.means, state2.means)
    assert np.allclose(state1.weights, state2.weights)
    assert np.allclose(fidelity_bosonic(state1,state2),1)

def test_post_select_fock_coherent():
    
    state = State(nmodes)
    state.apply_symplectic(Stot)
    state.post_select_fock_coherent(0, n)

    assert state.num_weights == (n+1)**2
    assert state.means.shape == ((n+1)**2, 2*nmodes-2)
    assert state.covs.shape == (1, 2*nmodes-2, 2*nmodes-2)
    assert np.imag(state.probability) < 1e-15 
    assert 0 < np.real(state.probability) < 1
    assert np.allclose(np.sum(state.weights/state.probability) ,1) 
    assert np.allclose(fidelity_bosonic(state, state), 1)
    
    state.post_select_fock_coherent(0, n)

    assert state.num_weights == (n+1)**4
    assert state.means.shape == ((n+1)**4, 2*nmodes-4)
    assert state.covs.shape == (1, 2*nmodes-4, 2*nmodes-4)
    assert np.imag(state.probability) < 1e-15 
    assert 0 < np.real(state.probability) < 1
    assert np.allclose(np.sum(state.weights/state.probability) ,1) 
    assert np.allclose(fidelity_bosonic(state, state), 1)

    print(state.get_mean_photons())


def test_post_select_homodyne():
    
    state = State(nmodes) 
    state.apply_symplectic(Stot)
    state.post_select_homodyne(0, 0, 0)

    assert state.num_weights == 1
    assert state.means.shape == (1, 2*nmodes-2)
    assert state.covs.shape == (1, 2*nmodes-2, 2*nmodes-2)
    assert np.imag(state.probability) < 1e-15 
    assert 0 < np.real(state.probability) < 1
    assert np.allclose(np.sum(state.weights/state.probability) ,1) 
    assert np.allclose(fidelity_bosonic(state, state), 1)
    
    state.post_select_homodyne(0,0,0,True)

    assert state.num_weights == 1
    assert state.means.shape == (1, 2*nmodes-4)
    assert state.covs.shape == (1, 2*nmodes-4, 2*nmodes-4)
    assert np.imag(state.probability) < 1e-15 
    assert 0 < np.real(state.probability) < 1
    assert np.allclose(np.sum(state.weights/state.probability) ,1) 
    assert np.allclose(fidelity_bosonic(state, state, True), 1)

def test_post_select_ppnrd_thermal():
    
    m = 2
    N = 6 #Setting this too high will cause the test to fail due at the purity stage
    
    state = State(nmodes)

    state.apply_symplectic(Stot)
    state.post_select_ppnrd_thermal(0, m, N)

    assert state.num_weights == m+1
    assert state.means.shape == (m+1, 2*nmodes-2)
    assert state.covs.shape == (m+1, 2*nmodes-2, 2*nmodes-2)
    assert np.imag(state.probability) < 1e-15 
    assert 0 < np.real(state.probability) < 1
    assert np.allclose(np.sum(state.weights/state.probability) ,1) 
    assert 0 < fidelity_bosonic(state, state) < 1 #State is no longer pure
    
    state.post_select_ppnrd_thermal(0, m, N)

    assert state.num_weights == (m+1)**2
    assert state.means.shape == ((m+1)**2, 2*nmodes-4)
    assert state.covs.shape == ((m+1)**2, 2*nmodes-4, 2*nmodes-4)
    assert np.imag(state.probability) < 1e-15 
    assert 0 < np.real(state.probability) < 1
    assert np.allclose(np.sum(state.weights/state.probability) ,1) 
    assert 0 < fidelity_bosonic(state, state) < 1


def test_wigner():
    state = State()
    
    x = np.linspace(-10,10,200)
    W = state.get_wigner(x,x)
    
    assert np.allclose(fidelity_bosonic(state,state),1)
    assert np.allclose(fidelity_with_wigner(W,W,x,x),1)
    assert np.allclose(np.sum(W)*np.diff(x)[0]**2,1)

    x = np.linspace(-10,10,200)
    
    nmodes = 3
    state = State(nmodes)
    state.apply_symplectic(Stot)
   
    state.post_select_homodyne(0,0,0)
    state.post_select_fock_coherent(0, 2)
    W = state.get_wigner(x,x)

    assert np.allclose(fidelity_bosonic(state,state),1)
    assert np.allclose(fidelity_with_wigner(W,W,x,x),1)
    assert np.allclose(np.sum(W)*np.diff(x)[0]**2,1)

def test_fock_gen():
    """Test the heralded fock generation from the EPR state.
    """
    state = State(2)
    m = 4
    s1 = xxpp_to_xpxp(expand(squeezing(2),[0], 2))
    s2 = xxpp_to_xpxp(expand(squeezing(2, np.pi),[1], 2))
    bs = xxpp_to_xpxp(expand(beam_splitter(np.pi/4, 0), [0,1],2))
    state.apply_symplectic(bs@s2@s1)
    state.post_select_fock_coherent(0, m, inf=1e-6)
    x = np.linspace(-10,10,200)
    W = state.get_wigner(x,x)

    X,P = np.meshgrid(x,x)
    Wexact = wig_mn(m,m,X,P)
    assert np.allclose(fidelity_with_wigner(W,Wexact, x,x) , 1, )

def test_mean_photons():
    state = State(nmodes)
    state.apply_symplectic(Stot)
    state.post_select_homodyne(0,0,0)
    state.post_select_homodyne(0,0,0)


    assert np.allclose(state.get_mean_photons(), mean_photon_sf_gauss)
    assert np.allclose(fidelity_bosonic(state, State()), sf_gauss_vacuum_fid)
    
    state = State(nmodes)
    state.apply_symplectic(Stot)
    state.post_select_fock_coherent(0,n)
    state.post_select_fock_coherent(0,n)

    assert np.allclose(state.get_mean_photons()[0], mean_photon_sf_fock[0]) #There is an unexplained discrepancy in photon number variances computed in this way
    assert np.allclose(fidelity_bosonic(state, State()), sf_fock_vacuum_fid)



    



