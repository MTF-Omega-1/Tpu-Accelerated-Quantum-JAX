import jax
import jax.numpy as jnp

def zero_state(num_qubits: int) -> jnp.ndarray:
    state = jnp.zeros(2 ** num_qubits, dtype=jnp.complex64)
    state = state.at[0].set(1.0 + 0j)
    return state.reshape((2,) * num_qubits)

def apply_gate(state: jnp.ndarray, gate_matrix: jnp.ndarray, targets: list[int]) -> jnp.ndarray:
    num_qubits = state.ndim
    k = len(targets)
    gate_tensor = gate_matrix.reshape((2,) * (2 * k))
    gate_contract_axes = list(range(k, 2 * k))
    state_contract_axes = list(targets)
    res = jnp.tensordot(gate_tensor, state, axes=(gate_contract_axes, state_contract_axes))
    uncontracted = [i for i in range(num_qubits) if i not in targets]
    inv_perm = [0] * num_qubits
    for i, t in enumerate(targets):
        inv_perm[t] = i
    for j, u in enumerate(uncontracted):
        inv_perm[u] = k + j
    return jnp.transpose(res, inv_perm)

def state_vector_flat(state: jnp.ndarray) -> jnp.ndarray:
    return state.reshape(-1)