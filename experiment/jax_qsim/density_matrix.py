"""
Density matrix simulator implemented in pure, differentiable JAX.
Supports noisy channels via Kraus operators and fully analytic gradients of noisy circuits.
"""

import jax
import jax.numpy as jnp
from . import gates
from .statevector import apply_gate as sv_apply_gate
from .observables import PauliString, Hamiltonian

def zero_state(num_qubits):
    """
    Returns the computational basis density matrix |00...0><00...0| 
    as a JAX array of shape (2,)*2N.
    """
    rho = jnp.zeros((2,) * (2 * num_qubits), dtype=jnp.complex64)
    zero_idx = (0,) * (2 * num_qubits)
    return rho.at[zero_idx].set(1.0)
