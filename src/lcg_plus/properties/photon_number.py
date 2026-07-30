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

def compute_photon_number_moments(State):
    """Calculate the first and second photon number moments of the state by summing over the weighted moments of each Gaussian,
    See Appendix: Photon-number moments of LCoGs https://arxiv.org/abs/2508.06175 
    """
    #Get rid of the hbar 
    covs = State.covs / State.hbar
    means = State.means /np.sqrt(State.hbar)
    cov_tr = np.trace(covs, axis1 = 1, axis2 = 2)
    cov_det = np.linalg.det(covs)
    
    mu_sq = np.einsum("...j,...j", means, means)

    exk = 1/2 *(cov_tr + mu_sq - 1)

    ex = np.exp(logsumexp(State.log_weights + np.log(exk))) / State.norm #first moment

    mucov = np.einsum("...j,...jk,...k", means, covs, means)

    vark = 1/2 * (cov_tr**2 - 2 * cov_det + 2 * mucov - 0.5) 

    var = np.exp(logsumexp(State.log_weights + np.log(vark))) / State.norm #second moment
    
    return ex, var