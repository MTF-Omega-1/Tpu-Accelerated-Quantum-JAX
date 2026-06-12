import jax.numpy as jnp
import pytest
import jax_qsim.ops as ops
from jax_qsim.core import zero_state, apply_gate, state_vector_flat

def test_unitary_gates():

    def is_unitary(matrix):
        dim = matrix.shape[0]
        identity = jnp.eye(dim, dtype=jnp.complex64)
        prod = jnp.dot(matrix, jnp.conj(matrix.T))
        return jnp.allclose(prod, identity, atol=1e-05)
    assert is_unitary(ops.H)
    assert is_unitary(ops.X)
    assert is_unitary(ops.Y)
    assert is_unitary(ops.Z)
    assert is_unitary(ops.S)
    assert is_unitary(ops.T)
    assert is_unitary(ops.CNOT)
    assert is_unitary(ops.CZ)
    assert is_unitary(ops.SWAP)
    assert is_unitary(ops.rx(0.5))
    assert is_unitary(ops.ry(-0.2))
    assert is_unitary(ops.rz(1.5))
    assert is_unitary(ops.phase_shift(0.8))
    assert is_unitary(ops.crx(0.3))
    assert is_unitary(ops.cry(-0.9))
    assert is_unitary(ops.crz(0.0))
    assert is_unitary(ops.cphase(1.2))

def test_apply_gates():
    state = zero_state(1)
    state = apply_gate(state, ops.H, [0])
    flat = state_vector_flat(state)
    expected = jnp.array([1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)], dtype=jnp.complex64)
    assert jnp.allclose(flat, expected)
    state = zero_state(1)
    state = apply_gate(state, ops.X, [0])
    flat = state_vector_flat(state)
    expected = jnp.array([0.0, 1.0], dtype=jnp.complex64)
    assert jnp.allclose(flat, expected)
    state = zero_state(2)
    state = apply_gate(state, ops.X, [0])
    state = apply_gate(state, ops.CNOT, [0, 1])
    flat = state_vector_flat(state)
    expected = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.complex64)
    assert jnp.allclose(flat, expected)
    state = zero_state(2)
    state = apply_gate(state, ops.X, [0])
    state = apply_gate(state, ops.SWAP, [0, 1])
    flat = state_vector_flat(state)
    expected = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.complex64)
    assert jnp.allclose(flat, expected)