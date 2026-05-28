"""
Quantum gates implemented in pure JAX.
All gates return complex64 JAX arrays.
"""

import jax.numpy as jnp

# ==============================================================================
# Non-parametric Gates (Single-qubit)
# ==============================================================================

def H():
    """Hadamard gate."""
    return jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(2.0)

def X():
    """Pauli-X (NOT) gate."""
    return jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)

def Y():
    """Pauli-Y gate."""
    return jnp.array([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex64)

def Z():
    """Pauli-Z gate."""
    return jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)

def S():
    """S (Phase) gate."""
    return jnp.array([[1.0, 0.0], [0.0, 1j]], dtype=jnp.complex64)

def T():
    """T gate (pi/8 gate)."""
    return jnp.array([[1.0, 0.0], [0.0, jnp.exp(1j * jnp.pi / 4.0)]], dtype=jnp.complex64)
