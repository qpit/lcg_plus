# bosonicplus
Simulator of continuous variable circuits with Gaussian components and photon number resolving detectors. This is an extension of the [bosonic backend](https://strawberryfields.ai/photonics/demos/run_intro_bosonic.html) from strawberryfields with significantly improved photon number resolving detector capabilities.

## Requirements
- See `pyproject.toml`

## Installation
`pip install -e .` from root.

## Features
- Simulation of phase-space Wigner functions in the Gaussian representation.
- Non-Gaussian states and measurements are represented as linear combinations of multivariate Gaussians.

### Main class

| Class | Description | Attributes | 
|-------|-------------|------------|
| `State(num_modes)` | CV state object (Wigner)  | `means`, `covs`, `weights`, `norm`, `num_k`, `hbar` (not implemented yet) |


| Method | Description | Parameters | 
|--------|-------------|------------|
| `update_data(new_data)` | Update the state with new data tuple | `new_data=[means, covs, weights]`  |
| `apply_symplectic(S)`| Applies a symplectic transform on the state | full `2Nx2N` symplectic matrix `S`|
| `apply_symplectic_fast(S, modes)` | Applies a symplectic transform on a subset of modes | smaller `2x2` or `4x4` symplectic matrix `S`, list of modes e.g. `modes=[0,1]` for a beamsplitter. Fa |
|`apply_displacement(d)`| Applies a displacement | `2Nx1` displacement vector, e.g. for a single mode `d=np.sqrt(2*hbar)*[alpha.real, alpha.imag]`
| `apply_loss(etas, nbars)` | Applies a multimode (thermal) loss channel |  transmissivities `etas`, thermal occupation numbers `nbars` |
| `post_select_fock_coherent(mode, n, inf=1e-4,red_gauss=True)` | Project a mode onto a fock state in the sum of gaussian representaion | mode number `mode`, photon number `n`, infidelity of the approximation `inf`, reduce the number of Gaussians with `red_gauss`. 
| `post_select_ppnrd_thermal(mode, n, M)` | Project a mode onto a pseudo-photon number detector (pPNRD) in sum of thermal state representation | mode number `mode`, click number `n`, number of on/off detectors in pPNRD `M` | 
| `post_select_homodyne(mode, angle, result, MP = False)` | Project a mode into a rotated position eigenstate | mode number `mode`, angle of quadrature `angle`, result of measurement `result`. For breeding of squeezed cats, higher precision may be required, so we can set `MP=True` to use `mpmath`|  
| `post_select_heterodyne(mode, alpha)` | Project a mode onto a coherent state | Not implemented | 
| `post_select_generaldyne(mode, r, alpha)`| Project a mode onto a squeezed coherent state | Not implemented |
|`get_wigner_bosonic(xvec,pvec,indices=None)`| Get the Wigner function of a single mode state, or the multivariate Wigner function over the specified indices | `xvec`, `pvec` |
| `multimode_copy(n)` | Copy a single mode state to `n` modes (tensor product) | |
|`add_state(state)` | Add a single mode state (tensor product) | Another instance of `bosonicplus.base.State` to add to the the current state | 

### Main functions

OBS: fidelity functions are probably just overlaps. Should resolve this ASAP. 

Imported from `bosonicplus.fidelity`:
| Function | Description | Comments | 
| ---------|-------------|------------|
| `fidelity_bosonic(state1,state2)` |  Calculate the quantum fidelity between two states in sum of Gaussians rep assuming one of them is pure.  | This might be actually be the overlap, double check |
| `fidelity_bosonic_new(state1,state2)` | Same as above, but for the reduced Gaussian formalism | Merge with above|
| `fidelity_with_wigner(W1,W2,xvec,pvec)` | Numerically calculate the overlap of the two Wigner functions, both evaluated in `xvec`, `pvec`. | 

Imported from `bosonicplus.states.nongauss`:
| Function | Description | Comments |
| ---------|-------------|------------|
|`prepare_fock_coherent(n, inf=1e-4)` | Make a Fock state in sum of Gauss rep | Maybe rename, since it sounds like its prepared from a protocol |
| `prepare_sqz_cat_coherent(r, alpha, k, MP =False)` | Make a squeezed cat, with possibility of increasing the precision of the weights with `mpmath`, because there is an issue with high `r` and `alpha` |  Same as above
| `prepare_gkp_coherent(n, which, N, inf = 1e-4)` | Make a `n` fock approx to a GKP state with GKP squeezing operator (Petr Marek paper) in linear comb of Gaussians rep | Same as above. See `bosonicplus.state.gkp_squeezing.py`


Imported from `bosonicplus.states.coherent`:
| Function | Description | Comments |
| ---------|-------------|------------|
|`gen_fock_superpos_coherent(coeffs, infid)`| Get the data tuple for the arbitrary fock superposition (pure) in sum of Gauss rep from a list of `coeffs` | 

Imported from `bosonicplus.states.from_sf`: Instance of `bosonicplus.base.State` filled with a state from strawberryfields bosonicbackend.
| Function | Description | Comments |
| ---------|-------------|------------|
| `prepare_gkp_bosonic`| | |
|`prepare_fock_bosonic`| | |
|`prepare_sqz_cat_bosonic`| | | 

See `bosonicplus.plotting` for plotting. Clean up and improve. 

### Wavefunction simulations (for superpositions of Gaussians, e.g. cats)
Reduces the number of Gaussians quadratically.

Imported from `bosonicplus.wavefunction`:

| Class | Description | Attributes | Comments |
| ---------|-------------|------------|----------|
| `PureBosonicState()` | CV pure state object  | `alphas`, `covs`, `coeffs` | In progress, there's a problem with Gaussian measurements | 




## Tutorial
See Jupyter notebooks in `src/bosonicplus/demos`.

