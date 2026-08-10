import pytest
import numpy as np
from lcg_plus.helper.matrices import symplectic_form, is_symplectic
from thewalrus.symplectic import xxpp_to_xpxp, xpxp_to_xxpp, sympmat
from thewalrus.random import random_symplectic
from thewalrus.symplectic import is_symplectic as is_symplectic_walrus

@pytest.mark.parametrize("num_modes",[100,10])
def test_symplecitc_form(num_modes : int):
    #Check the symplectic form
    omg = symplectic_form(num_modes)
    assert np.allclose(xpxp_to_xxpp(omg), sympmat(num_modes)), "The symplectic form is not equal to the symplectic form from thewalrus"

@pytest.mark.parametrize("num_modes",[100,10])
def test_is_symplectic(num_modes : int):
    #Generate a random symplectic matrix and check testing functions
    S = random_symplectic(num_modes)
    assert is_symplectic_walrus(S), "is_symplectic from thewalrus failed"
    assert is_symplectic(xxpp_to_xpxp(S)), "is_symplectic failed"