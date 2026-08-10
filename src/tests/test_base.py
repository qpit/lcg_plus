
import pytest
import numpy as np
from scipy.stats import unitary_group
from lcg_plus.base import State
from thewalrus.random import random_symplectic
from thewalrus.symplectic import xxpp_to_xpxp, is_symplectic


#Test Gaussian states
@pytest.mark.parametrize("num_modes", [10,5])
def test_initialize(num_modes  : int):
    state = State(num_modes)
    assert state.means.shape == (1, 2*num_modes)
    assert state.covs.shape == (1, 2*num_modes, 2*num_modes)


@pytest.mark.parametrize("num_modes", [10,5])
def test_gaussian(num_modes : int):

    S = random_symplectic(num_modes)
    assert is_symplectic(S)
    S = xxpp_to_xpxp(S)

    d = np.random.uniform(size=(2*num_modes)) #random displacement

    state = State(num_modes)
    state.apply_displacement(d)
    state.apply_gaussian_unitary(S)

    r = S @ d
    V = state.hbar/2 * S @ S.T

    assert np.allclose(state.means, r), "symplectic transform on means failed"
    assert np.allclose(state.covs, V), "symplectic transform on covs failed"

    X = np.random.uniform(size=(2*num_modes,2*num_modes))
    Y = np.random.uniform(size=(2*num_modes,2*num_modes))
    state.apply_gaussian_channel(X, Y)
    assert np.allclose(state.means, X @ r),  "gausian channel on means failed"
    assert np.allclose(state.covs, X @ V @ X.T + Y) ,"gausian channel on covs failed"




#@pytest.mark.parameterize("num_modes",[10,5]):
#def test_post_select


