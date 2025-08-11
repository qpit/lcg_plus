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
from scipy.special import factorial, genlaguerre

## To do: speed up, e.g. rho is self-adjoint.

def Dmn(alpha : complex, m : int, n : int) -> complex:
    """Calculates mn'th element of complex displacement operator with alpha using the Cahill1969 formula
    """
    
    
    if n > m:
        m, n = n, m
        alpha = -alpha.conjugate()

        
    prefactor = np.sqrt(factorial(n)/factorial(m)) * alpha**(m-n) * np.exp(-np.abs(alpha)**2 / 2)
    dmn = prefactor * genlaguerre(n, m-n)(np.abs(alpha)**2)

    return dmn


def density_mn(alpha : complex, cutoff : int) -> np.ndarray:
    """Construct Fock decomposition of the displacement operator up to a cutoff
    """
    if cutoff == 0:
        rho = Dmn(alpha, 0,0)
        
    else:
        rho = np.zeros((cutoff+1, cutoff+1), dtype = 'complex')
        
        for i in np.arange(cutoff+1):
            for j in np.arange(cutoff+1):
                rho[i,j] = Dmn(alpha, i,j)
    return rho