import pytest
import numpy as np
from lcg_plus.operations.channels import is_valid_gaussian_channel, loss_channel_matrices, gain_channel_matrices

@pytest.mark.parametrize("num_modes", [10])
def test_loss_channel_matrices(num_modes : int):
    etas = np.random.uniform(size=num_modes)
    nbars = 2*np.random.uniform(size=num_modes)
    X, Y = loss_channel_matrices(etas, nbars)
    assert is_valid_gaussian_channel(X,Y)

@pytest.mark.parametrize("num_modes", [10])
def test_gain_channel_matrices(num_modes : int):
    gains = 1+np.random.uniform(size=num_modes)
    X, Y = gain_channel_matrices(gains)
    assert is_valid_gaussian_channel(X,Y)

