"""
JAX Quantum Circuit Simulation Suite (jax_qsim)

A research-level, high-performance, differentiable quantum simulator in pure JAX.
Optimized for XLA compilation on CPU and CUDA GPU (RTX 2050).
"""

from .circuit import Circuit
from .statevector import Statevector
from .density_matrix import DensityMatrix
from .observables import PauliString, Hamiltonian
from . import gates

__all__ = [
    'Circuit',
    'Statevector',
    'DensityMatrix',
    'PauliString',
    'Hamiltonian',
    'gates',
]
