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
from scipy.special import logsumexp

def compute_quadrature_moments(State):
    """Compute the first and second moment of the quadrature operators.
    """
    #first moment
    ex = np.real_if_close(np.exp(logsumexp(State.log_weights[:,np.newaxis] + np.log(State.means), axis = 0))/ State.norm)
    
    if State.num_k != State.num_weights:
        ex = ex.real

    #second moment
    offset = np.tensordot(ex, ex, axes = 0)
        
    var_tilde = State.covs + np.einsum("...j,...k", State.means, State.means)

    cov = np.exp(logsumexp(State.log_weights[:,np.newaxis, np.newaxis] + np.log(var_tilde), axis = 0)) / State.norm
        
    var = np.real_if_close(cov - offset)

    return ex, var
                