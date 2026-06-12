import jax
import jax.numpy as jnp
from jax_qsim import Circuit
from jax_qsim.observables import PauliString, Hamiltonian, expectation
import jax_qsim.noise as noise

def test_circuit_compilation():
    c = Circuit(num_qubits=3)
    c.h(0)
    c.cnot(0, 1)
    c.cnot(1, 2)
    run_compiled = c.compile()
    params = jnp.array([])
    state = run_compiled(params)
    expected = jnp.zeros((2, 2, 2), dtype=jnp.complex64)
    expected = expected.at[0, 0, 0].set(1.0 / jnp.sqrt(2.0))
    expected = expected.at[1, 1, 1].set(1.0 / jnp.sqrt(2.0))
    assert jnp.allclose(state, expected)

def test_hamiltonian_expectation():
    c = Circuit(num_qubits=2)
    c.h(0)
    c.cnot(0, 1)
    state = c.run(jnp.array([]))
    obs1 = PauliString({0: 'Z', 1: 'Z'})
    obs2 = PauliString({0: 'X', 1: 'X'})
    h = Hamiltonian([1.0, 0.5], [obs1, obs2])
    val = h.expectation(state)
    assert jnp.allclose(val, 1.5)

def test_noise_channel():
    state = jnp.zeros((2,), dtype=jnp.complex64).at[0].set(1.0)
    kraus = noise.depolarizing_channel(0.2)
    key = jax.random.PRNGKey(42)
    new_state, new_key = noise.apply_channel(state, kraus, [0], key)
    norm = jnp.sum(jnp.abs(new_state) ** 2)
    assert jnp.allclose(norm, 1.0)
    assert not jnp.allclose(key, new_key)