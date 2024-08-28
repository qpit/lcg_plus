import thewalrus.symplectic as symp
import numpy as np

# GAUSSIAN OPERATIONS
# ------------------------------------
def beamsplitter(theta, phi, k, l, n):
    r"""Beam splitter symplectic between modes k and l for n mode system.
    Args:
        theta (float): real beamsplitter angle
        phi (float): complex beamsplitter angle
        k (int): first mode
        l (int): second mode
        n (int): total number of modes
    Raises:
        ValueError: if the first mode equals the second mode
    Returns:
        bs (array): n mode symplectic matrix in xpxp notation
    """

    if k == l:
        raise ValueError("Cannot use the same mode for beamsplitter inputs.")

    bs = symp.expand(symp.beam_splitter(theta, phi), [k, l], n)
    bs = symp.xxpp_to_xpxp(bs) #convert to xpxp notation

    return bs
    
def squeeze(r, phi, k, n):
    r"""Symplectic for squeezing mode ``k`` by the amount ``r``.
    Args:
        r (float): squeezing magnitude
        k (int): mode to be squeezed
        n (int): total number of modes
    Returns:
        sq (array): n mode squeezing symplectic matrix in xpxp notation
    """

    sq = symp.expand(symp.squeezing(r,phi), k, n)
    sq = symp.xxpp_to_xpxp(sq)

    return sq
    
def rotation(theta, k, n):
    r"""Symplectic for a rotation operation on mode k.
        Args:
            theta (float): rotation angle
            k (int): mode to be rotated
            n (int): total number of modes
        Returns:
            sr (array): n mode rotation symplectic matrix in xpxp notation
    """
    sr = symp.expand(symp.rotation(theta), k, n)
    sr = symp.xxpp_to_xpxp(sr)
    return sr

def unitary_xp2a(U):
    """Convert 2Nx2N unitary matrix in the xxxppp representation to a 
    NxN unitary matrix operating on annihilation operators.
    Where did I get this from?
    """
    n = U.shape[0] // 2
    In = np.identity(n)
    T = .5 * np.block([[In, 1j*In], [In, -1j*In]])
    Ua = T @ U @ np.linalg.inv(T)
    return Ua[:n, :n]


def symmetric_multiport_interferometer_unitary(N):
    """Unitary for a symmetric multiport interferometer
    from Eq. 18-19 of https://journals.aps.org/pra/pdf/10.1103/PhysRevA.55.2564

    Equal probability of each input mode going to N output ports
    
    Args:
        N (int): number of modes
    Returns:
        U ((N,N) complex array): unitary
    """
    U = np.zeros((N,N), dtype = 'complex')
    gamma_N = np.exp(1j * 2 * np.pi/N)

    for i in np.arange(N):
        for j in np.arange(N):
            U[i,j] = 1/np.sqrt(N) * gamma_N ** (i*j)
    return U

def symmetric_multiport_interferometer_symplectic(N):
    """Symplectic for a symmetric multiport interferometer

        S = [[X, -Y],[Y, X]] where X = U.real and Y = U.imag where U is the unitary matrix
    
    Args:
        N (int): number of modes
    Returns:
        S ((2N,2N) array): symplectic matrix in xpxp
    """
    S = symp.interferometer(symmetric_multiport_interferometer_unitary(N)) #in xxpp
    return symp.xxpp_to_xpxp(S) #to xpxp
