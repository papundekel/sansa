########################################################################################################################
#
# CORE FACTORIZATION OPERATIONS
#
# The implementation is based on https://github.com/pymatting/pymatting/blob/master/pymatting/preconditioner/ichol.py,
# but the mathematical algorithm is different:
# We use a modification of the icfm algorithm by Lin and More: https://epubs.siam.org/doi/abs/10.1137/S1064827597327334
#
########################################################################################################################
import ctypes
import logging

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

libsansa = np.ctypeslib.load_library("libsansa", loader_path="./")
libsansa.core_icf.argtypes = [
    ctypes.c_int64,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int64,
    ctypes.c_float,
]
libsansa.core_icf.restype = ctypes.c_int64


def icf(
    A: sp.csc_matrix,
    l2: float,
    max_nnz: int,
    shift_step: float = 1e-3,
    shift_multiplier: float = 2.0,
) -> sp.csc_matrix:
    if isinstance(A, sp.csr_matrix):
        A = A.T
    if not isinstance(A, sp.csc_matrix):
        raise ValueError("Matrix A must be a scipy.sparse.csc_matrix")
    m, n = A.shape
    assert m == n, f"A must be square, got shape {A.shape}"
    # need at least 1 element per column
    # otherwise it doesn't make sense (mathematically, and the factorization algorithm would fail)
    if max_nnz < n:
        max_nnz = n
    Lv = np.empty(max_nnz, dtype=np.float32)  # Values of non-zero elements of L
    Lr = np.empty(max_nnz, dtype=np.int64)  # Row indices of non-zero elements of L
    Lp = np.zeros(n + 1, dtype=np.int64)  # Start(Lp[i]) and end(Lp[i+1]) index of L[:, i] in Lv
    shift = np.float32(l2)
    counter = -1
    nnz = np.int64(-1)
    while nnz == -1:
        counter += 1
        nnz = libsansa.core_icf(n, A.data, A.indices, A.indptr, Lv, Lr, Lp, max_nnz, shift)
        # if shift is too small, increase it
        if nnz == -1:
            next_shift = l2 + shift_step * (shift_multiplier**counter)
            logger.info(
                f"""
                Incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A 
                with L2={shift:.4e}. Continuing with L2={next_shift:.4e}.
                """
            )
            shift = next_shift
    Lv = Lv[:nnz]
    Lr = Lr[:nnz]
    return sp.csc_matrix((Lv, Lr, Lp), (n, n))
