import numpy as np

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def has_even_dimensions(matrix):
    m, n = matrix.shape
    
    if n != m:
        return False
    if n % 2 != 0:
        return False

def symplectic_form(num_modes):
    Omega = np.array([[0,1],[-1,0]]) #Single mode 
    return np.kron(np.eye(num_modes),Omega)

def is_symplectic(mat, rtol=1e-05, atol=1e-08):
    """Check if mat is symplectic"""
    
    has_even_dimensions(mat)
    num_modes = mat.shape[0] // 2
    Omega = symplectic_form(num_modes)
    return np.allclose(Omega, mat @ Omega @ mat.T, rtol=rtol, atol=atol)

def is_valid_covmat(covmat, hbar = 2, rtol = 1e-05, atol=1e-08):
    """Check if covmat is a valid quantum covariance matrix.
    See https://github.com/XanaduAI/thewalrus/blob/master/thewalrus/quantum/gaussian_checks.py#L26
    """

    #Check dimensions
    has_even_dimensions(covmat)
    #Check if symmetric
    if not np.allclose(covmat, covmat.T, rtol=rtol, atol=atol):
        return False
    
    #Check for violation of uncertainty relations
    num_modes = covmat.shape[0] // 2
    Omega = symplectic_form(num_modes)
    vals = np.linalg.eigvalsh(covmat + 0.5j * hbar * Omega)
    vals[np.abs(vals)< atol] = 0.0
    if np.all(vals >=0):
        return True


