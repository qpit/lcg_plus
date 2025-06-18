def expand(S, modes, N):
    r"""Expands a Symplectic matrix S to act on the entire subsystem.
    If the input is a single mode symplectic, then extends it to act
    on multiple modes.

    Supports scipy sparse matrices. Instances of ``coo_array``, ``dia_array``,
    ``bsr_array`` will be transformed into `csr_array``.

    Args:
        S (ndarray or spmatrix): a :math:`2M\times 2M` Symplectic matrix
        modes (Sequence[int]): the list of modes S acts on
        N (int): full size of the subsystem

    Returns:
        array: the resulting :math:`2N\times 2N` Symplectic matrix
    """
    M = S.shape[0] // 2
    S2 = (
        np.identity(2 * N, dtype=S.dtype)
        if not issparse(S)
        else sparse_identity(2 * N, dtype=S.dtype, format="csr")
    )

    if issparse(S) and isinstance(S, (coo_array, dia_array, bsr_array)):
        # cast to sparse matrix that supports slicing and indexing
        warnings.warn(
            "Unsupported sparse matrix type, returning a Compressed Sparse Row (CSR) matrix."
        )
        S = csr_array(S)

    w = np.array([modes]) if isinstance(modes, int) else np.array(modes)

    # extend single mode symplectic to act on selected modes
    if M == 1:
        for m in w:
            S2[m, m], S2[m + N, m + N] = S[0, 0], S[1, 1]  # X, P
            S2[m, m + N], S2[m + N, m] = S[0, 1], S[1, 0]  # XP, PX
        return S2

    # make symplectic act on the selected subsystems
    S2[w.reshape(-1, 1), w.reshape(1, -1)] = S[:M, :M].copy()  # X
    S2[(w + N).reshape(-1, 1), (w + N).reshape(1, -1)] = S[M:, M:].copy()  # P
    S2[w.reshape(-1, 1), (w + N).reshape(1, -1)] = S[:M, M:].copy()  # XP
    S2[(w + N).reshape(-1, 1), w.reshape(1, -1)] = S[M:, :M].copy()  # PX
    return S2





[docs]
def expand_vector(alpha, mode, N, hbar=2.0):
    """Returns the phase-space displacement vector associated to a displacement.

    Args:
        alpha (complex): complex displacement
        mode (int): mode index
        N (int): number of modes

    Returns:
        array: phase-space displacement vector of size 2*N
    """
    alpharealdtype = np.dtype(type(alpha))

    r = np.zeros(2 * N, dtype=alpharealdtype)
    r[mode] = np.sqrt(2 * hbar) * alpha.real
    r[N + mode] = np.sqrt(2 * hbar) * alpha.imag
    return r