"""Shared low-level NumPy memory layout helpers."""

import numpy as np


def _flat_memory_view(a):
    """Return a copy-free flat view in the physical order of C/F arrays."""
    if a.flags.f_contiguous and not a.flags.c_contiguous:
        flat = a.T.reshape(-1)
    elif a.flags.c_contiguous:
        flat = a.reshape(-1)
    else:
        raise ValueError("Array is not C- or F-contiguous")
    if not np.shares_memory(flat, a):
        raise RuntimeError("Contiguous flattening unexpectedly created a copy")
    return flat
