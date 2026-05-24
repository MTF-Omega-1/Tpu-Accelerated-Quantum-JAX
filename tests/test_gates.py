import jax.numpy as jnp
import pytest
import jax_qsim.ops as ops
from jax_qsim.core import zero_state, apply_gate, state_vector_flat

def test_unitary_gates():
    """Verify that all standard gates are indeed unitary matrices."""
    def is_unitary(matrix):
        dim = matrix.shape[0]
        identity = jnp.eye(dim, dtype=jnp.complex64)
        # U * U^dag = I
        prod = jnp.dot(matrix, jnp.conj(matrix.T))
        return jnp.allclose(prod, identity, atol=1e-5)

    # Single-qubit fixed gates
    assert is_unitary(ops.H)
    assert is_unitary(ops.X)
    assert is_unitary(ops.Y)
    assert is_unitary(ops.Z)
    assert is_unitary(ops.S)
    assert is_unitary(ops.T)
    
    # Multi-qubit fixed gates
    assert is_unitary(ops.CNOT)
    assert is_unitary(ops.CZ)
    assert is_unitary(ops.SWAP)
    
    # Parameterized gates
    assert is_unitary(ops.rx(0.5))
    assert is_unitary(ops.ry(-0.2))
    assert is_unitary(ops.rz(1.5))
    assert is_unitary(ops.phase_shift(0.8))
    
    # Controlled parameterized gates
    assert is_unitary(ops.crx(0.3))
    assert is_unitary(ops.cry(-0.9))
    assert is_unitary(ops.crz(0.0))
    assert is_unitary(ops.cphase(1.2))

def test_apply_gates():
    """Verify application of single and multi-qubit gates on zero-state."""
    # 1. H|0> = |+> = 1/sqrt(2) (|0> + |1>)
    state = zero_state(1)
    state = apply_gate(state, ops.H, [0])
    flat = state_vector_flat(state)
    expected = jnp.array([1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)], dtype=jnp.complex64)
    assert jnp.allclose(flat, expected)

    # 2. X|0> = |1>
    state = zero_state(1)
    state = apply_gate(state, ops.X, [0])
    flat = state_vector_flat(state)
    expected = jnp.array([0.0, 1.0], dtype=jnp.complex64)
    assert jnp.allclose(flat, expected)

    # 3. CNOT |10> = |11> (in qubit 0, 1 indexing where qubit 0 is control and 1 is target)
    state = zero_state(2)
    # Apply X to qubit 0 -> |10>
    state = apply_gate(state, ops.X, [0])
    # Apply CNOT from 0 to 1 -> |11>
    state = apply_gate(state, ops.CNOT, [0, 1])
    flat = state_vector_flat(state)
    # |11> corresponds to flat index 3 (since binary 11 = 3 in decimal)
    expected = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.complex64)
    assert jnp.allclose(flat, expected)

    # 4. SWAP |10> = |01>
    state = zero_state(2)
    state = apply_gate(state, ops.X, [0])  # |10>
    state = apply_gate(state, ops.SWAP, [0, 1])  # |01>
    flat = state_vector_flat(state)
    # |01> is binary 01, which is flat index 1 (under qubit 0, 1 indexing where 1 is the least significant qubit)
    # Wait, in index-dimension mapping:
    # State shape is (2, 2).
    # If state = |10>, it is element [1, 0], which reshapes to flat index 2.
    # After swap, state is |01>, element [0, 1], which reshapes to flat index 1.
    expected = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.complex64)
    assert jnp.allclose(flat, expected)
