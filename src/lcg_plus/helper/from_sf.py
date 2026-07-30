import numpy as np

# Helper functions from strawberryfields
# ---------------------------------------
def chop_in_blocks_multi(m, id_to_delete):
    r"""
    Splits an array of (symmetric) matrices each into 3 blocks (``A``, ``B``, ``C``).

    Blocks ``A`` and ``C`` are diagonal blocks and ``B`` is the offdiagonal block.

    Args:
        m (ndarray): array of matrices
        id_to_delete (ndarray): array for the indices that go into ``C``

    Returns:
        tuple: tuple of the ``A``, ``B`` and ``C`` matrices
    """
    A = np.delete(m, id_to_delete, axis=1)
    A = np.delete(A, id_to_delete, axis=2)
    B = np.delete(m[:, :, id_to_delete], id_to_delete, axis=1)
    C = m[:, id_to_delete, :][:, :, id_to_delete]
    return (A, B, C)


def chop_in_blocks_multi_v2(m, id_to_delete):
    r"""
    Splits an array of (symmetric) matrices each into 3 blocks (``A``, ``B``, ``C``).

    Blocks ``A`` and ``C`` are diagonal blocks and ``B`` is the offdiagonal block.

    Args:
        m (ndarray): array of matrices
        id_to_delete (ndarray): array for the indices that go into ``C``

    Returns:
        tuple: tuple of the ``A``, ``B`` and ``C`` matrices
    """
    A = np.delete(m, id_to_delete, axis=2)
    A = np.delete(A, id_to_delete, axis=3)
    B = np.delete(m[:, :, :, id_to_delete], id_to_delete, axis=2)
    C = m[:, :, id_to_delete, :][:, :, :, id_to_delete]
    return (A, B, C)


def chop_in_blocks_vector_multi(v, id_to_delete):
    r"""
    For an array of vectors ``v``, splits ``v`` into two arrays of vectors,
    ``va`` and ``vb``. ``vb`` contains the components of ``v`` specified by
    ``id_to_delete``, and ``va`` contains the remaining components.

    Args:
        v (ndarray): array of vectors
        id_to_delete (ndarray): array for the indices that go into vb

    Returns:
        tuple: tuple of ``(va,vb)`` vectors
    """
    
    id_to_keep = np.sort(list(set(np.arange(len(v[0]))) - set(id_to_delete)))
    
    va = v[:, id_to_keep]
    vb = v[:, id_to_delete]
    return (va, vb)

def chop_in_blocks_vector_multi_v2(v, id_to_delete):
    r"""
    For an array of vectors ``v``, splits ``v`` into two arrays of vectors,
    ``va`` and ``vb``. ``vb`` contains the components of ``v`` specified by
    ``id_to_delete``, and ``va`` contains the remaining components.

    Args:
        v (ndarray): array of vectors
        id_to_delete (ndarray): array for the indices that go into vb

    Returns:
        tuple: tuple of ``(va,vb)`` vectors
    """
    
    id_to_keep = np.sort(list(set(np.arange(len(v[0,0]))) - set(id_to_delete)))
    
    va = v[:, :, id_to_keep]
    vb = v[:, :, id_to_delete]
    return (va, vb)