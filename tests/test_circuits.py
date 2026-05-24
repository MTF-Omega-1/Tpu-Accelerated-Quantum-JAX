import jax
import jax.numpy as jnp
from jax_qsim import Circuit
from jax_qsim.observables import PauliString, Hamiltonian, expectation
import jax_qsim.noise as noise

def test_circuit_compilation():
    """Verify that compiling a circuit and running it yields correct GHZ state."""
    # 3-qubit GHZ state circuit
    c = Circuit(num_qubits=3)
    c.h(0)
    c.cnot(0, 1)
    c.cnot(1, 2)
    
    # Compile the circuit
    run_compiled = c.compile()
    
    # Execute (no parameters required)
    params = jnp.array([])
    state = run_compiled(params)
    
    # Expected: (|000> + |111>) / sqrt(2)
    expected = jnp.zeros((2, 2, 2), dtype=jnp.complex64)
    expected = expected.at[0, 0, 0].set(1.0 / jnp.sqrt(2.0))
    expected = expected.at[1, 1, 1].set(1.0 / jnp.sqrt(2.0))
    
    assert jnp.allclose(state, expected)

def test_hamiltonian_expectation():
    """Verify calculating expectation value of a composite Hamiltonian on Bell state."""
    c = Circuit(num_qubits=2)
    c.h(0)
    c.cnot(0, 1)
    
    state = c.run(jnp.array([]))
    
    # H = 1.0 * Z0*Z1 + 0.5 * X0*X1
    obs1 = PauliString({0: 'Z', 1: 'Z'})
    obs2 = PauliString({0: 'X', 1: 'X'})
    
    h = Hamiltonian([1.0, 0.5], [obs1, obs2])
    
    # expectation:
    # <Bell| Z0*Z1 |Bell> = 1.0
    # <Bell| X0*X1 |Bell> = 1.0
    # Total expectation = 1.0*1.0 + 0.5*1.0 = 1.5
    val = h.expectation(state)
    assert jnp.allclose(val, 1.5)

def test_noise_channel():
    """Verify noisy channel application via quantum trajectories."""
    state = jnp.zeros((2,), dtype=jnp.complex64).at[0].set(1.0) # |0>
    kraus = noise.depolarizing_channel(0.2)
    key = jax.random.PRNGKey(42)
    
    new_state, new_key = noise.apply_channel(state, kraus, [0], key)
    
    # 1. State vector must remain normalized
    norm = jnp.sum(jnp.abs(new_state) ** 2)
    assert jnp.allclose(norm, 1.0)
    
    # 2. Key must be updated
    assert not jnp.allclose(key, new_key)
